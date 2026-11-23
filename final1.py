import argparse
import codecs
import csv
import html
import io
import json
import os
import re
import sys
import textwrap
import unicodedata
import warnings
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.sparse import vstack
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, normalize

warnings.filterwarnings("ignore")

# La consola puede no ser UTF-8 (Windows): así nunca truena al imprimir emojis.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================
# 0. UTILIDADES DE CONSOLA
# ============================================================

def titulo(texto, nivel=1):
    if nivel == 1:
        print("\n" + "=" * 100)
        print(texto)
        print("=" * 100)
    else:
        print(f"\n--- {texto} ---")


def recortar(s, n=220):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def barra(v, vmax, ancho=24):
    if vmax is None or vmax <= 0 or not np.isfinite(v):
        return ""
    return "█" * int(round(ancho * max(v, 0) / vmax))


def mostrar_tabla(df, **kw):
    print(df.to_string(**kw))


LOG = []


def paso(nombre, antes, despues, nota=""):
    """Registra cada etapa del procesamiento (bitácora)."""
    LOG.append({"paso": nombre, "filas_antes": antes,
                "filas_despues": despues, "nota": nota})


# ============================================================
# 1. TEXTO: ACENTOS, STOPWORDS, LEMATIZACIÓN
# ============================================================

@lru_cache(maxsize=300000)
def sin_acentos(s):
    """Quita acentos SOLO en alfabeto latino (no daña cirílico, árabe, hindi...)."""
    if s.isascii():
        return s
    nfd = unicodedata.normalize("NFD", s)
    letras = [c for c in nfd if unicodedata.category(c).startswith("L")]
    if letras and not all(unicodedata.name(c, "").startswith("LATIN")
                          for c in letras):
        return s
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Listas básicas incluidas (por si NLTK no está instalado). Se usan sin acentos.
_SW = {
    "es": "a al algo algun alguna algunas alguno algunos ante antes aqui asi aun aunque "
          "cada como con contra cual cuales cuando de del desde donde dos e el ella ellas "
          "ellos en entre era eran eres es esa esas ese eso esos esta estaba estaban "
          "estamos estan estar estas este esto estos estoy fue fueron fui ha habia han "
          "hasta hay he hemos la las le les lo los mas me mi mis mismo mucha muchas mucho "
          "muchos muy no nos nosotros nuestra nuestro nuestros o os otra otras otro otros "
          "para pero por porque pues que quien quienes se sea ser si siempre sido sino "
          "sobre somos son soy su sus tambien tanto te tengo tiene tienen todas todo todos "
          "tu tus un una uno unos usted ustedes vez voy y ya yo ser estar haber tener "
          "hacer ir poder deber",
    "en": "a about above after again all am an and any are as at be because been before "
          "being below between both but by can could did do does doing down during each "
          "few for from further had has have having he her here hers herself him himself "
          "his how i if in into is it its itself just me more most my myself no nor not of "
          "off on once only or other our ours out over own same she should so some such "
          "than that the their theirs them then there these they this those through to too "
          "under until up very was we were what when where which while who whom why will "
          "with would you your yours yourself",
    "pt": "a ao aos as com como da das de dela dele deles depois do dos e ela elas ele "
          "eles em entre era essa esse esta este eu foi ha isso isto ja la lhe mais mas me "
          "mesmo meu minha muito na nas nao nem no nos nossa nosso num numa o os ou para "
          "pela pelas pelo pelos por qual quando que quem se seu sua so tambem te tem um "
          "uma voce voces",
    "fr": "a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur "
          "lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui "
          "sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y "
          "est sont etait ete etre avoir",
    "de": "aber alle als also am an auch auf aus bei bin bis bist da dann das dass dein "
          "dem den der des die dies diese dieser doch du ein eine einem einen einer er es "
          "euch euer fuer hat hatte hier ich ihr im in ist ja kann mein mit nach nicht "
          "noch nun nur oder ohne sehr sein sich sie sind so und uns unser von war waren "
          "was weil wenn wir wird wo zu zum zur",
    "it": "a ad al alla allo anche che chi ci come con cui da dal della dei del di e ed "
          "era essere gli ha ho i il in io la le lei lo loro lui ma mi mia mio nei nel "
          "nella no non noi o per perche piu quel quella quello questa questo se si sono "
          "su sua suo tra tu tuo un una uno vi voi",
}
BUILTIN_SW = {k: {sin_acentos(w) for w in v.split()} for k, v in _SW.items()}

# Palabras que se conservan porque pueden cambiar el sentido de una reseña
CONSERVAR_ES = {"ni", "sin", "nada", "nunca", "tampoco", "poco"}

NLTK_NOMBRES = {
    "es": "spanish", "en": "english", "pt": "portuguese", "fr": "french",
    "de": "german", "it": "italian", "nl": "dutch", "ru": "russian",
    "sv": "swedish", "da": "danish", "no": "norwegian", "fi": "finnish",
    "tr": "turkish", "ar": "arabic", "ro": "romanian", "hu": "hungarian",
    "id": "indonesian", "el": "greek", "ca": "catalan",
}


@lru_cache(maxsize=None)
def stopwords_idioma(lang):
    """Stopwords del idioma: lista incluida + NLTK si ya está instalado.
    (Nunca descarga nada: el script no escribe en disco.)"""
    base = set(BUILTIN_SW.get(lang, ()))
    if lang in NLTK_NOMBRES:
        try:
            from nltk.corpus import stopwords
            base |= {sin_acentos(w.casefold())
                     for w in stopwords.words(NLTK_NOMBRES[lang])}
        except Exception:
            pass
    if lang == "es":
        base -= {sin_acentos(w) for w in CONSERVAR_ES}
    return frozenset(base)


try:
    import simplemma as _simplemma
    HAY_SIMPLEMMA = True
except Exception:
    _simplemma = None
    HAY_SIMPLEMMA = False

# Nombres propios que un lematizador suele estropear (Orizaba -> "orizar")
NO_LEMATIZAR = {"orizaba", "lagos", "puebla", "poblano", "poblana"}


@lru_cache(maxsize=None)
def lema(token, lang):
    """Lema palabra por palabra (con caché). Si no hay simplemma o el idioma no
    está soportado, devuelve el token tal cual."""
    if not HAY_SIMPLEMMA or lang in ("und", "") or token in NO_LEMATIZAR:
        return token
    try:
        return _simplemma.lemmatize(token, lang=lang).casefold()
    except Exception:
        return token


# ============================================================
# 2. LIMPIEZA ROBUSTA (cualquier idioma, hashtags, emojis, URLs...)
# ============================================================

RE_URL = re.compile(r"(?:https?://|ftp://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
                    re.I)
RE_RT = re.compile(r"(?<!\w)RT\s+(?=@)")
RE_MENCION = re.compile(r"(?<!\w)@\w+")
RE_HASHTAG = re.compile(r"(?<!\w)[#＃](\w+)")
RE_INVISIBLES = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
RE_EMOJI = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF"
    "\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\u2190-\u21FF\u2900-\u297F"
    "\u3030\u303D\u3297\u3299\u203C\u2049\u2122\u2139\u24C2\u25AA-\u25FE"
    "\ufe0f\u20e3]")
RE_EMOT_POS = re.compile(r"(?<!\w)[:;=][\-^o']?[)\]dDpP}](?!\w)|<3|(?<!\w)\^[_.\-]?\^(?!\w)")
RE_EMOT_NEG = re.compile(r"(?<!\w)[:;=]['\-^o]?[(\[{](?!\w)|(?<!\w)>:[(\[]|(?<!\w)T[_.\-]T(?!\w)")
RE_MONEDA = re.compile(
    r"[$€£¥]\s?\d[\d.,]*|\d[\d.,]*\s?(?:pesos|mxn|usd|eur|dolares|dólares|euros)\b",
    re.I)
# Letras de cualquier alfabeto (+ marcas combinantes de hindi, árabe, tailandés...)
RE_TOKEN = re.compile(
    r"(?:[^\W\d_]|[\u0300-\u036f\u0591-\u05c7\u0610-\u061a\u064b-\u065f"
    r"\u0900-\u0dff\u0e31-\u0e3a\u0e47-\u0e4e])+")
# Idiomas sin espacios entre palabras (chino, japonés, coreano, tailandés): bigramas
RE_SIN_ESPACIOS = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u0e00-\u0e7f]+")

POS_EMOJI = ("SMIL", "GRINN", "HEART", "THUMBS UP", "SPARKLES", "CLAPPING", "PARTY",
             "STAR-STRUCK", "HEAVY CHECK", "HUNDRED POINTS", "WINKING", "BEAMING",
             "KISS", "HUGGING", "RELIEVED", "SAVORING", "RAINBOW", "TROPHY", "MEDAL",
             "CELEBRATION", "TEARS OF JOY", "GLOWING STAR", "WHITE MEDIUM STAR",
             "BLACK STAR", "FIREWORK", "SUN WITH")
NEG_EMOJI = ("ANGRY", "ENRAGED", "POUTING", "CRYING", "SOBBING", "DISAPPOINTED",
             "WEARY", "UNAMUSED", "NAUSEATED", "VOMITING", "THUMBS DOWN",
             "BROKEN HEART", "PILE OF POO", "CONFOUNDED", "FEARFUL", "ANGUISHED",
             "FROWNING", "PENSIVE", "SYMBOLS ON MOUTH", "SKULL", "SICK")


@lru_cache(maxsize=None)
def valor_emoji(ch):
    """+1 / -1 / 0 según el nombre Unicode del emoji (sin librerías externas)."""
    nombre = unicodedata.name(ch, "").upper()
    if not nombre:
        return 0
    if "TEARS OF JOY" in nombre:
        return 1
    if any(k in nombre for k in NEG_EMOJI):
        return -1
    if any(k in nombre for k in POS_EMOJI):
        return 1
    return 0


def _dividir_camel(s):
    """#ViajaMexico -> 'Viaja Mexico' (sirve para hashtags en cualquier alfabeto)."""
    out = []
    for i, c in enumerate(s):
        if i and c.isupper() and s[i - 1].islower():
            out.append(" ")
        out.append(c)
    return "".join(out)


def _bigramas_cjk(m):
    run = m.group(0)
    if len(run) == 1:
        return f" {run} "
    return " " + " ".join(run[i:i + 2] for i in range(len(run) - 1)) + " "


def limpiar_texto(t):
    """Devuelve (tokens, emo_valor, n_emojis, n_hashtags).

    tokens: palabras en minúscula, sin URLs/menciones/números/puntuación/emojis.
    """
    t = html.unescape(str(t))
    t = unicodedata.normalize("NFKC", t)
    t = RE_INVISIBLES.sub("", t)
    t = RE_URL.sub(" ", t)
    t = RE_RT.sub(" ", t)
    t = RE_MENCION.sub(" ", t)
    n_hash = len(RE_HASHTAG.findall(t))
    t = RE_HASHTAG.sub(lambda m: " " + _dividir_camel(m.group(1)) + " ", t)

    # Emojis y emoticonos: se cuentan para el sentimiento y se quitan del texto
    emojis = RE_EMOJI.findall(t)
    emo = sum(valor_emoji(c) for c in emojis)
    emo += len(RE_EMOT_POS.findall(t)) - len(RE_EMOT_NEG.findall(t))
    n_emo = len(emojis)
    t = RE_EMOJI.sub(" ", t)
    t = RE_EMOT_POS.sub(" ", t)
    t = RE_EMOT_NEG.sub(" ", t)

    t = RE_MONEDA.sub(" dinero ", t)             # montos -> palabra genérica
    t = re.sub(r"n['’`]t\b", " not", t, flags=re.I)   # don't -> do not
    t = t.casefold()
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)         # holaaaa -> holaa
    t = RE_SIN_ESPACIOS.sub(_bigramas_cjk, t)
    tokens = RE_TOKEN.findall(t)
    tokens = [x for x in tokens if len(x) > 1 or RE_SIN_ESPACIOS.fullmatch(x)]
    emo = float(np.clip(emo, -3, 3))
    return tokens, emo, n_emo, n_hash


# ============================================================
# 3. DETECCIÓN DE IDIOMA (sin modelos de lenguaje)
# ============================================================

try:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    HAY_LANGDETECT = True
except Exception:
    HAY_LANGDETECT = False

_SCRIPT_A_IDIOMA = (("CYRILLIC", "ru"), ("ARABIC", "ar"), ("HIRAGANA", "ja"),
                    ("KATAKANA", "ja"), ("HANGUL", "ko"), ("CJK", "zh"),
                    ("THAI", "th"), ("DEVANAGARI", "hi"), ("GREEK", "el"),
                    ("HEBREW", "he"))


@lru_cache(maxsize=None)
def _alfabeto(c):
    nombre = unicodedata.name(c, "")
    for pref, cod in _SCRIPT_A_IDIOMA:
        if nombre.startswith(pref):
            return cod
    return "lat" if nombre.startswith("LATIN") else "otro"


def detectar_idioma(tokens):
    """Idioma de un comentario: 'es', 'en', 'pt', 'fr', 'de', 'it', ... o 'und'."""
    if not tokens:
        return "und"
    texto = " ".join(tokens)
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return "und"
    cuenta = Counter(_alfabeto(c) for c in letras)
    alf, n_alf = cuenta.most_common(1)[0]
    if alf not in ("lat", "otro") and n_alf / len(letras) >= 0.5:
        return alf
    if HAY_LANGDETECT and len(texto) >= 20 and len(tokens) >= 4:
        try:
            cand = detect_langs(texto)[0]
            if cand.prob >= 0.6:
                return cand.lang.split("-")[0]
        except Exception:
            pass
    # Respaldo: traslape con stopwords de los idiomas incluidos
    toks = {sin_acentos(t) for t in tokens}
    pun = {lg: len(toks & sw) for lg, sw in BUILTIN_SW.items()}
    mejor = max(pun.values())
    if mejor == 0:
        return "und"
    ganadores = [lg for lg, v in pun.items() if v == mejor]
    return ganadores[0] if len(ganadores) == 1 else "und"


# ============================================================
# 4. SENTIMIENTO: calificación (si existe) o léxico + emojis
# ============================================================
# Léxicos pequeños por idioma (sin acentos al cargarlos). Sin embeddings.

_LEX = {
    "es": ("bueno buena buenos buenas excelente excelentes hermoso hermosa hermosos "
           "hermosas bonito bonita bonitos bonitas lindo linda lindos lindas increible "
           "increibles maravilloso maravillosa maravillosos maravillosas fantastico "
           "fantastica espectacular genial perfecto perfecta perfectos perfectas encanta "
           "encanto encantan encantado encantada encantador encantadora amo amamos amor "
           "recomiendo recomendable recomendado recomendamos gracias feliz felices "
           "contento contenta satisfecho satisfecha agradable agradables limpio limpia "
           "tranquilo tranquila rico rica deliciosa delicioso deliciosas deliciosos "
           "sabroso sabrosa amable amables atento atenta rapido rapida mejor mejores "
           "magnifico magnifica impresionante divertido divertida chido chida favorito "
           "favorita disfrute disfrutamos disfrutar acogedor acogedora comodo comoda "
           "barato barata economico economica gratis bien paraiso sorprendente precioso "
           "preciosa preciosos preciosas padrisimo bellisimo bellisima",
           "malo mala malos malas pesimo pesima horrible horribles terrible terribles "
           "decepcion decepcionante decepcionado decepcionada desagradable sucio sucia "
           "sucios sucias caro cara caros caras carisimo costoso costosa lento lenta "
           "peligro peligroso peligrosa inseguro insegura ruido ruidoso ruidosa fatal "
           "feo fea feos feas aburrido aburrida abandonado abandonada estafa estafaron "
           "robo robaron ladron ladrones abusivo abusivos grosero grosera groseros queja "
           "quejas problema problemas peor peores engano enganan enganaron timo odio odie "
           "molesto molesta basura asco asqueroso asquerosa deficiente deplorable "
           "lamentable pobre mediocre incomodo incomoda trampa trampas fraude saturado "
           "saturada abarrotado sobrevalorado sobrevalorada"),
    "en": ("good great excellent amazing wonderful beautiful lovely nice fantastic "
           "perfect love loved loving recommend recommended best awesome enjoy enjoyed "
           "enjoyable clean friendly helpful delicious tasty stunning gorgeous charming "
           "cozy comfortable safe worth happy pleasant fun breathtaking magical lively "
           "relaxing peaceful superb outstanding impressive affordable cheap free "
           "incredible favorite favourite thanks",
           "bad terrible horrible awful poor worst dirty disgusting rude slow expensive "
           "overpriced overrated disappointing disappointed disappointment boring noisy "
           "crowded unsafe dangerous scam rip ripoff waste mediocre unfriendly "
           "unpleasant avoid hate hated broken smelly filthy mess"),
    "pt": ("bom boa bons boas otimo otima excelente lindo linda lindos lindas "
           "maravilhoso maravilhosa incrivel perfeito perfeita adorei adoro amei amo "
           "recomendo recomendado gostei gostoso gostosa delicioso deliciosa limpo "
           "limpa tranquilo tranquila agradavel simpatico simpatica barato fantastico "
           "espetacular encantador",
           "ruim ruins pessimo pessima horrivel terrivel sujo suja caro cara caros "
           "caras decepcionante decepcionado decepcao lento lenta perigoso perigosa "
           "barulhento fraco fraca odiei odeio feio feia golpe chato chata "
           "desagradavel mediocre superfaturado"),
    "fr": ("bon bonne bons bonnes excellent excellente magnifique superbe merveilleux "
           "merveilleuse beau belle beaux belles genial incroyable parfait parfaite "
           "adore aime recommande agreable propre calme charmant charmante delicieux "
           "delicieuse sympa sympathique gentil gentille accueillant formidable "
           "fantastique heureux heureuse",
           "mauvais mauvaise horrible terrible nul nulle decevant decevante decu decue "
           "deception sale cher chere chers cheres lent lente dangereux dangereuse "
           "bruyant arnaque desagreable moche ennuyeux mediocre pire"),
    "de": ("gut gute guter gutes guten super toll wunderbar wunderschoen wunderschon "
           "schoen schon herrlich perfekt ausgezeichnet fantastisch empfehle "
           "empfehlenswert empfohlen liebe liebevoll freundlich sauber ruhig lecker "
           "angenehm genial traumhaft zufrieden begeistert beeindruckend preiswert "
           "guenstig kostenlos",
           "schlecht schlechte schlechter schlimm schrecklich furchtbar schmutzig "
           "dreckig teuer ueberteuert enttaeuschend enttaeuscht langsam gefaehrlich "
           "laut unfreundlich langweilig abzocke betrug mittelmaessig ekelhaft"),
    "it": ("buono buona buoni buone ottimo ottima eccellente bellissimo bellissima "
           "bello bella meraviglioso meravigliosa fantastico fantastica perfetto "
           "perfetta adoro amo consiglio consigliato consigliatissimo pulito pulita "
           "tranquillo tranquilla delizioso deliziosa gentile cordiale piacevole "
           "stupendo stupenda splendido splendida economico incredibile",
           "cattivo cattiva pessimo pessima orribile terribile sporco sporca caro cara "
           "cari care deludente deluso delusione lento lenta pericoloso pericolosa "
           "rumoroso fregatura truffa scadente mediocre antipatico noioso sgradevole"),
}
LEXICO = {lg: (frozenset(sin_acentos(w) for w in p.split()),
               frozenset(sin_acentos(w) for w in n.split()))
          for lg, (p, n) in _LEX.items()}
LEXICO_TODOS = (frozenset().union(*[v[0] for v in LEXICO.values()]),
                frozenset().union(*[v[1] for v in LEXICO.values()]))

_NEGADORES = {
    "es": "no nunca jamas ni tampoco sin",
    "en": "not no never neither nor without cannot",
    "pt": "nao nunca jamais nem sem",
    "fr": "ne pas jamais sans aucun",
    "de": "nicht kein keine keinen nie niemals ohne",
    "it": "non mai senza nessun",
}
NEGADORES = {lg: frozenset(sin_acentos(w) for w in v.split())
             for lg, v in _NEGADORES.items()}
NEGADORES_TODOS = frozenset().union(*NEGADORES.values())


def _en_lexico(t, conj):
    if t in conj:
        return True
    if len(t) > 4 and t.endswith("s") and t[:-1] in conj:
        return True
    return len(t) > 5 and t.endswith("es") and t[:-2] in conj


def puntaje_lexico(tokens, lang, emo):
    """Puntaje bruto de polaridad: palabras del léxico (con negación) + emojis."""
    pos, neg = LEXICO.get(lang, LEXICO_TODOS)
    negadores = NEGADORES.get(lang, NEGADORES_TODOS)
    toks = [sin_acentos(t) for t in tokens]
    total = 0.0
    for i, t in enumerate(toks):
        v = 1 if _en_lexico(t, pos) else (-1 if _en_lexico(t, neg) else 0)
        if v == 0:
            continue
        if any(x in negadores for x in toks[max(0, i - 3):i]):
            v = -v
        total += v
    return total + emo


# ============================================================
# 5. CARGA DE CUALQUIER ARCHIVO
# ============================================================

NOMBRES_TEXTO = ["texto", "text", "comentario", "comentarios", "comment", "comments",
                 "review", "reviews", "resena", "resenas", "contenido", "content",
                 "body", "caption", "descripcion", "description", "mensaje",
                 "message", "tweet", "post", "opinion", "feedback", "publicacion",
                 "selftext", "full_text", "reviewtext", "review_text"]
NOMBRES_FUENTE = ["fuente", "source", "plataforma", "platform", "origen", "sitio",
                  "red", "red_social", "canal", "dataset"]
NOMBRES_CALIF = ["calificacion", "rating", "stars", "star", "estrellas", "puntuacion",
                 "puntaje", "valoracion", "nota", "bubbles", "reviewrating", "target"]


def _norm(c):
    return sin_acentos(str(c)).lower().strip().replace(" ", "_")


def leer_texto_crudo(ruta):
    """Lee bytes y decodifica probando varias codificaciones (nunca falla)."""
    with open(ruta, "rb") as f:
        raw = f.read()
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        import chardet
        enc = chardet.detect(raw[:300000])["encoding"]
        if enc:
            return raw.decode(enc, errors="replace")
    except Exception:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _json_a_tabla(texto):
    try:
        obj = json.loads(texto)
    except Exception:
        obj = []
        for linea in texto.splitlines():          # JSON Lines
            linea = linea.strip()
            if not linea:
                continue
            try:
                obj.append(json.loads(linea))
            except Exception:
                return None
    if isinstance(obj, dict):
        listas = [v for v in obj.values() if isinstance(v, list)]
        obj = listas[0] if listas else [obj]
    if not isinstance(obj, list) or not obj:
        return None
    if all(isinstance(x, str) for x in obj):
        return pd.DataFrame({"texto": obj})
    try:
        return pd.json_normalize(obj).astype(str)
    except Exception:
        return None


def _detectar_delimitador(texto, ext):
    if ext == ".tsv":
        return "\t"
    muestra = [l for l in texto.splitlines() if l.strip()][:60]
    if len(muestra) < 2:
        return None
    mejor, mejor_n = None, 0
    for d in (",", ";", "\t", "|"):
        try:
            filas = list(csv.reader(muestra, delimiter=d))
        except csv.Error:
            continue
        cuentas = [len(f) for f in filas]
        moda, frec = Counter(cuentas).most_common(1)[0]
        if moda >= 2 and frec / len(cuentas) >= 0.8 and moda > mejor_n:
            mejor, mejor_n = d, moda
    return mejor


def _lineas_a_tabla(texto):
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    # Párrafos si los comentarios ocupan varias líneas separadas por líneas en blanco
    if len(parrafos) >= 5 and len(lineas) / len(parrafos) > 1.5:
        return pd.DataFrame({"texto": [" ".join(p.split()) for p in parrafos]})
    return pd.DataFrame({"texto": lineas})


def leer_archivo(ruta):
    """Devuelve una tabla (DataFrame de strings) a partir de cualquier archivo."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls", ".ods"):
        hojas = pd.read_excel(ruta, sheet_name=None, dtype=str)
        return max(hojas.values(), key=len).fillna("")
    texto = leer_texto_crudo(ruta)
    if ext in (".json", ".jsonl", ".ndjson") or texto.lstrip()[:1] in ("[", "{"):
        df = _json_a_tabla(texto)
        if df is not None:
            return df
    delim = _detectar_delimitador(texto, ext)
    if delim is None and ext == ".csv":
        delim = ","
    if delim:
        try:
            df = pd.read_csv(io.StringIO(texto), sep=delim, engine="python",
                             dtype=str, on_bad_lines="skip", keep_default_na=False)
            if df.shape[1] >= 1 and len(df) > 0:
                return df
        except Exception:
            pass
    return _lineas_a_tabla(texto)


def _buscar_columna(df, pedida, candidatas):
    norm = {_norm(c): c for c in df.columns}
    if pedida:
        if pedida in df.columns:
            return pedida
        if _norm(pedida) in norm:
            return norm[_norm(pedida)]
        raise SystemExit(f"No encuentro la columna '{pedida}'. "
                         f"Columnas disponibles: {list(df.columns)}")
    for n in candidatas:
        if n in norm:
            return norm[n]
    return None


def _elegir_col_texto(df, pedida):
    col = _buscar_columna(df, pedida, NOMBRES_TEXTO)
    if col is not None and df[col].astype(str).str.strip().ne("").mean() > 0.3:
        return col
    mejor, mejor_len = None, 0
    for c in df.columns:                       # heurística: la columna más "texto"
        s = df[c].astype(str)
        if pd.to_numeric(s, errors="coerce").notna().mean() > 0.8:
            continue
        largo = s.str.len().mean()
        if largo > mejor_len:
            mejor, mejor_len = c, largo
    return mejor


def _calificacion_normalizada(df, excluir):
    col = _buscar_columna(df, None, NOMBRES_CALIF)
    if col is None or col in excluir:
        return None
    s = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False),
                      errors="coerce")
    if s.notna().mean() < 0.3 or s.min() < 0 or s.max() > 10 or s.nunique() < 2:
        return None
    escala = 5.0 if s.max() <= 5 else 10.0
    return s / escala


def cargar_fuentes(rutas, col_texto=None, col_fuente=None):
    """Une todos los archivos en una sola tabla: texto, fuente, calif_norm."""
    partes, usadas = [], Counter()
    for ruta in rutas:
        if not os.path.isfile(ruta):
            raise SystemExit(f"No existe el archivo: {ruta}")
        tabla = leer_archivo(ruta)
        ctexto = _elegir_col_texto(tabla, col_texto)
        if ctexto is None:
            print(f"AVISO: no pude identificar la columna de texto en {ruta}; se omite.")
            continue
        etiqueta = os.path.splitext(os.path.basename(ruta))[0]
        usadas[etiqueta] += 1
        if usadas[etiqueta] > 1:
            etiqueta = f"{etiqueta}_{usadas[etiqueta]}"
        cfuente = _buscar_columna(tabla, col_fuente, NOMBRES_FUENTE) \
            if (col_fuente or len(rutas) == 1) else None
        if cfuente is not None and 2 <= tabla[cfuente].nunique() <= 40:
            fuente = tabla[cfuente].astype(str).str.strip()
        else:
            fuente = pd.Series(etiqueta, index=tabla.index)
        califs = _calificacion_normalizada(tabla, {ctexto})
        parte = pd.DataFrame({
            "texto": tabla[ctexto].astype(str),
            "fuente": fuente.values,
            "calif_norm": califs.values if califs is not None else np.nan})
        partes.append(parte)
        print(f"  · {os.path.basename(ruta)}: {len(parte):,} filas | columna de texto: "
              f"'{ctexto}'" + (" | con calificación" if califs is not None else ""))
    if not partes:
        raise SystemExit("No se pudo leer texto de ningún archivo.")
    return pd.concat(partes, ignore_index=True)


def elegir_dmo(fuentes, pedida):
    fuentes = list(fuentes)
    if pedida:
        q = sin_acentos(pedida).lower()
        hits = [f for f in fuentes if q in sin_acentos(f).lower()]
        if not hits:
            raise SystemExit(f"--dmo '{pedida}' no coincide con ninguna fuente: {fuentes}")
        if len(hits) > 1:
            print(f"AVISO: '{pedida}' coincide con varias fuentes; se usa '{hits[0]}'.")
        return hits[0]
    auto = [f for f in fuentes if re.search(r"sectur|dmo", f, re.I)]
    if auto:
        return auto[0]
    if sys.stdin and sys.stdin.isatty():
        print("\nFuentes detectadas:")
        for i, f in enumerate(fuentes, 1):
            print(f"  {i}. {f}")
        while True:
            r = input("¿Cuál es la DMO? (número o nombre): ").strip()
            if r.isdigit() and 1 <= int(r) <= len(fuentes):
                return fuentes[int(r) - 1]
            hits = [f for f in fuentes if r and r.lower() in f.lower()]
            if hits:
                return hits[0]
    print(f"AVISO: no se indicó la DMO; se toma la primera fuente: '{fuentes[0]}'.")
    return fuentes[0]


# ============================================================
# 6. PREPROCESAMIENTO (paso 1 del proyecto)
# ============================================================

def preprocesar(df_raw, A):
    titulo("1. LIMPIEZA Y PREPROCESAMIENTO DE TEXTO")
    df = df_raw.copy()
    n0 = len(df)
    paso("Carga de archivos", n0, n0,
         f"{df['fuente'].nunique()} fuentes: {', '.join(map(str, df['fuente'].unique()))}")

    df["texto_original"] = df["texto"].astype(str).str.strip()
    vacio = df["texto_original"].str.lower().isin(["", "nan", "none", "null", "n/a"])
    antes = len(df)
    df = df[~vacio].reset_index(drop=True)
    paso("Eliminación de textos nulos/vacíos", antes, len(df))

    antes = len(df)
    df = df.drop_duplicates(subset=["fuente", "texto_original"]).reset_index(drop=True)
    paso("Duplicados exactos (misma fuente y mismo texto)", antes, len(df))

    if A.max_por_fuente and A.max_por_fuente > 0:
        antes = len(df)
        rng = np.random.default_rng(A.semilla)
        mantener = []
        for _, pos in df.groupby("fuente").indices.items():
            pos = np.asarray(pos)
            if len(pos) > A.max_por_fuente:
                pos = rng.choice(pos, A.max_por_fuente, replace=False)
            mantener.extend(pos.tolist())
        df = df.iloc[sorted(mantener)].reset_index(drop=True)
        if len(df) < antes:
            paso("Muestreo por fuente (tope de comentarios)", antes, len(df),
                 f"máx. {A.max_por_fuente} por fuente")

    # --- limpieza: URLs, menciones, hashtags, emojis, números, alargamientos ---
    res = [limpiar_texto(t) for t in df["texto_original"]]
    df["tokens"] = [r[0] for r in res]
    df["emo_valor"] = [r[1] for r in res]
    df["n_emojis"] = [r[2] for r in res]
    df["n_hashtags"] = [r[3] for r in res]
    paso("Limpieza (URLs, @, #, emojis, números, símbolos, minúsculas)",
         len(df), len(df),
         f"{int((df['n_emojis'] > 0).sum())} comentarios con emojis; "
         f"{int((df['n_hashtags'] > 0).sum())} con hashtags")

    # --- idioma por comentario ---
    if A.idioma != "auto":
        df["idioma"] = A.idioma                     # idioma seleccionado por el usuario
        if A.idioma not in BUILTIN_SW and A.idioma not in NLTK_NOMBRES:
            print(f"Aviso: no hay stopwords incluidas para '{A.idioma}'; solo se limpian "
                  "símbolos y se lematiza si simplemma lo soporta.")
    else:
        df["idioma"] = [detectar_idioma(t) for t in df["tokens"]]
    conocidos = df.loc[df["idioma"] != "und", "idioma"].value_counts(normalize=True)
    dominantes = [lg for lg, p in conocidos.items() if p >= 0.10][:3] or ["es", "en"]
    sw_und = frozenset().union(*[stopwords_idioma(lg) for lg in dominantes])
    paso("Detección de idioma por comentario" if A.idioma == "auto"
         else "Idioma seleccionado (sin detección automática)", len(df), len(df),
         (("langdetect + " if HAY_LANGDETECT else "stopwords + alfabeto; ")
          if A.idioma == "auto" else f"--idioma {A.idioma}; ")
         + ", ".join(f"{k}={v}" for k, v in df["idioma"].value_counts().head(6).items()))

    # --- sentimiento bruto (antes de quitar stopwords/negaciones) ---
    df["puntaje_bruto"] = [puntaje_lexico(t, lg, e) for t, lg, e
                           in zip(df["tokens"], df["idioma"], df["emo_valor"])]

    # --- stopwords + lematización ---
    def limpiar_tokens(tokens, lg):
        sw = sw_und if lg == "und" else stopwords_idioma(lg)
        salida = []
        for t in tokens:
            if sin_acentos(t) in sw:
                continue
            l = sin_acentos(lema(t, lg))
            if l in sw or (len(l) < 2 and not RE_SIN_ESPACIOS.fullmatch(l)):
                continue
            salida.append(l)
        return salida

    df["tokens_limpios"] = [limpiar_tokens(t, lg)
                            for t, lg in zip(df["tokens"], df["idioma"])]
    df["texto_limpio"] = df["tokens_limpios"].str.join(" ")
    paso("Tokenización, stopwords por idioma y "
         + ("lematización (simplemma)" if HAY_SIMPLEMMA else "sin lematización (instala simplemma)"),
         len(df), len(df),
         f"{int(df['tokens'].str.len().sum())} -> {int(df['tokens_limpios'].str.len().sum())} tokens")

    antes = len(df)
    df = df[df["tokens_limpios"].str.len() > 0].reset_index(drop=True)
    paso("Eliminación de comentarios vacíos tras la limpieza", antes, len(df),
         "p. ej. solo emojis, solo URLs o solo stopwords")

    print("\nBitácora del preprocesamiento:")
    mostrar_tabla(pd.DataFrame(LOG), index=False,
                  justify="left", max_colwidth=90)

    print("\nComentarios por fuente (la etiqueta de fuente se conserva en todo el pipeline):")
    resumen = df.groupby("fuente").agg(
        comentarios=("texto_original", "size"),
        tokens_medios=("tokens_limpios", lambda s: round(s.str.len().mean(), 1)),
        con_emojis=("n_emojis", lambda s: int((s > 0).sum())),
        con_hashtags=("n_hashtags", lambda s: int((s > 0).sum())))
    resumen["idiomas"] = df.groupby("fuente")["idioma"].apply(
        lambda s: ", ".join(f"{k}:{v}" for k, v in s.value_counts().head(4).items()))
    mostrar_tabla(resumen)

    if not HAY_LANGDETECT:
        print("\nNota: sin 'langdetect' el idioma se estima con stopwords (6 idiomas) y "
              "alfabeto. Para mayor precisión: pip install langdetect")
    if not HAY_SIMPLEMMA:
        print("Nota: sin 'simplemma' no se lematiza. Para lematizar: pip install simplemma")
    return df


# ============================================================
# 7. TF-IDF GLOBAL (lexical, sin embeddings), ISOLATION FOREST Y N-GRAMAS  (paso 2)
# ============================================================

def construir_tfidf(df):
    n = len(df)
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2),
                          min_df=2 if n >= 300 else 1, max_df=0.85, sublinear_tf=True)
    try:
        X = vec.fit_transform(df["texto_limpio"])
    except ValueError:                       # vocabulario vacío
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=1)
        X = vec.fit_transform(df["texto_limpio"])
    return vec, X.tocsr()


def tabla_ngramas(textos, n, top=15):
    try:
        cv = CountVectorizer(ngram_range=(n, n), token_pattern=r"\S+", lowercase=False)
        M = cv.fit_transform(textos)
    except ValueError:
        return pd.DataFrame(columns=["ngrama", "frecuencia", "n_comentarios"])
    t = pd.DataFrame({"ngrama": cv.get_feature_names_out(),
                      "frecuencia": np.asarray(M.sum(0)).ravel(),
                      "n_comentarios": np.asarray((M > 0).sum(0)).ravel()})
    return (t.sort_values(["frecuencia", "n_comentarios"], ascending=False)
             .head(top).reset_index(drop=True))


def imprimir_ngramas(textos, etiqueta, top=15):
    print(f"\n  >> {etiqueta}  ({len(textos)} comentarios)")
    if len(textos) == 0:
        print("     (sin comentarios)")
        return
    for n, nombre in ((1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")):
        t = tabla_ngramas(textos, n, top)
        print(f"\n     {nombre}:")
        if t.empty:
            print("       (sin n-gramas)")
            continue
        vmax = t["frecuencia"].max()
        for _, r in t.iterrows():
            print(f"       {r['ngrama']:<32} {int(r['frecuencia']):>5}  "
                  f"({int(r['n_comentarios'])} coment.)  {barra(r['frecuencia'], vmax, 18)}")


CARACT_IF = {
    "log_tokens": "longitud (log de tokens limpios)",
    "log_caracteres": "longitud (log de caracteres originales)",
    "diversidad_lexica": "diversidad léxica (tokens únicos / tokens)",
    "sim_centroide": "similitud coseno con el centroide de su fuente",
    "idf_medio": "rareza media del vocabulario (IDF medio)",
    "emojis": "número de emojis",
    "hashtags": "número de hashtags",
}


def _caracteristicas_if(df, X, vec):
    """Variables numéricas INTERPRETABLES por comentario (sin embeddings) para IsolationForest.
    El TF-IDF solo aporta la similitud con el centroide de la fuente y el IDF medio."""
    n = len(df)
    ntok = df["tokens_limpios"].str.len().to_numpy(dtype=float)
    nchar = df["texto_original"].str.len().to_numpy(dtype=float)
    ttr = np.array([len(set(t)) / len(t) if len(t) else 0.0 for t in df["tokens_limpios"]])
    sim = np.zeros(n)
    for _, idx in df.groupby("fuente").indices.items():
        idx = np.asarray(idx)
        cen = np.asarray(X[idx].mean(axis=0)).ravel()
        nrm = np.linalg.norm(cen)
        if nrm > 0:
            sim[idx] = np.asarray(X[idx] @ (cen / nrm)).ravel()
    idf = getattr(vec, "idf_", np.ones(X.shape[1]))
    B = (X > 0).astype(np.float64).tocsr()
    nnz = np.maximum(np.asarray(B.sum(axis=1)).ravel(), 1)
    idf_medio = np.asarray(B @ idf).ravel() / nnz
    return pd.DataFrame({
        "log_tokens": np.log1p(ntok), "log_caracteres": np.log1p(nchar),
        "diversidad_lexica": ttr, "sim_centroide": sim, "idf_medio": idf_medio,
        "emojis": df["n_emojis"].to_numpy(float), "hashtags": df["n_hashtags"].to_numpy(float)}, index=df.index)



# ============================================================
# PALETA, AISLAMIENTO VECTORIAL, N-GRAMAS Y NUBES DE PALABRAS
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
         "#D55E00", "#56B4E9", "#F0E442", "#999999"]
C_DMO, H_DMO = "#0072B2", "///"
C_PERC, H_PERC = "#E69F00", "..."
C_POS, H_POS = "#009E73", "\\\\"
C_NEG, H_NEG = "#D55E00", "xx"
MARCAS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

# Paleta continua de los mapas de calor (se elige con --paleta)
PALETAS_CONTINUAS = ("viridis", "plasma", "inferno", "magma", "cividis")
CMAP = "cividis"


def _color_texto(valor_norm):
    """Texto blanco u oscuro según la luminosidad de la paleta en ese valor (0-1)."""
    r, g, b, _ = plt.get_cmap(CMAP)(float(np.clip(valor_norm, 0, 1)))
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"


def graficar_resumen_limpieza(df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    conteo = df.groupby("fuente").size().sort_values()
    ax[0].barh(conteo.index, conteo.values,
               color=[OKABE[i % len(OKABE)] for i in range(len(conteo))],
               edgecolor="black", hatch="///")
    ax[0].set_title("Comentarios válidos por fuente")
    ax[0].set_xlabel("Comentarios")
    for y, v in enumerate(conteo.values):
        ax[0].text(v, y, f" {v}", va="center")
    longitudes = df.assign(tokens=df["tokens_limpios"].str.len())
    sns.boxplot(data=longitudes, x="fuente", y="tokens", ax=ax[1],
                color="#56B4E9", showfliers=True)
    ax[1].set_title("Longitud de comentarios (tokens limpios)")
    ax[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()


def _graficar_proyeccion_2d_3d(Z, flags, titulo_base, semilla=42):
    """Proyecta un espacio TF-IDF estandarizado para visualizarlo (no para detectar)."""
    n_muestras, n_features = Z.shape
    if n_muestras < 3 or n_features < 2:
        print(f"No se puede generar la proyección 2D/3D de {titulo_base}: "
              f"se requieren al menos 3 registros y 2 características.")
        return

    componentes_2d = min(2, n_features)
    coords2 = TruncatedSVD(n_components=componentes_2d,
                           random_state=semilla).fit_transform(Z)
    flags = np.asarray(flags, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords2[~flags, 0], coords2[~flags, 1], s=24,
               c=C_DMO, marker="o", alpha=.48, label="Normales")
    if flags.any():
        ax.scatter(coords2[flags, 0], coords2[flags, 1], s=85,
                   c=C_NEG, marker="X", edgecolors="black", linewidths=.6,
                   label="Atípicos", zorder=3)
    ax.set_title(f"{titulo_base} — proyección 2D (SVD)")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()

    if n_muestras >= 4 and n_features >= 3:
        coords3 = TruncatedSVD(n_components=3,
                               random_state=semilla).fit_transform(Z)
        fig = plt.figure(figsize=(11, 8))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.scatter(coords3[~flags, 0], coords3[~flags, 1], coords3[~flags, 2],
                    s=22, c=C_DMO, marker="o", alpha=.38, label="Normales")
        if flags.any():
            ax3.scatter(coords3[flags, 0], coords3[flags, 1], coords3[flags, 2],
                        s=85, c=C_NEG, marker="X", edgecolors="black",
                        linewidths=.6, label="Atípicos", depthshade=False)
        ax3.set_title(f"{titulo_base} — proyección 3D (SVD)")
        ax3.set_xlabel("Componente 1")
        ax3.set_ylabel("Componente 2")
        ax3.set_zlabel("Componente 3")
        ax3.legend()
        fig.tight_layout()
    else:
        print(f"No se genera la proyección 3D de {titulo_base}: "
              "se requieren al menos 4 registros y 3 características.")


def detectar_isolation_forest_vectorial(df, A):
    """Detecta por fuente con Isolation Forest sobre TF-IDF estandarizado.

    Isolation Forest trabaja en el espacio vectorial TF-IDF completo, después de
    StandardScaler(with_mean=False). SVD se usa exclusivamente para visualizar
    las observaciones en 2D y 3D; nunca se usa para entrenar el detector.
    """
    titulo("2. DETECCIÓN DE ATÍPICOS: ISOLATION FOREST EN ESPACIO TF-IDF ESTANDARIZADO")
    vec, X = construir_tfidf(df)
    df["outlier"] = False
    df["if_score"] = 0.0
    df["if_motivo"] = ""
    matrices = {}
    resumen = []

    print("Representación: TF-IDF de unigramas y bigramas")
    print("Estandarización: StandardScaler(with_mean=False), ajustado por fuente")
    print("Detección: Isolation Forest en el espacio vectorial completo")
    print("Visualización: SVD 2D/3D (solo proyección; no participa en la detección)")
    print(f"Isolation Forest: {A.arboles} árboles; contaminación={A.contaminacion:.1%}")

    nombres = vec.get_feature_names_out()
    for fuente, indices in df.groupby("fuente", sort=False).indices.items():
        indices = np.asarray(indices)
        X_fuente = X[indices]

        # Estandarizar todas las fuentes para que todas aparezcan en las gráficas
        # generales, incluso cuando alguna tenga menos registros que --min-if.
        escalador = StandardScaler(with_mean=False)
        Z = escalador.fit_transform(X_fuente).tocsr()
        matrices[fuente] = (indices, Z)

        if len(indices) < A.min_if:
            resumen.append({
                "fuente": fuente, "comentarios": len(indices), "atípicos": 0,
                "% atípicos": 0.0,
                "nota": f"No se aplicó IF: menos de {A.min_if} comentarios"
            })
            print(f"\n{fuente}: se omite Isolation Forest; solo hay {len(indices)} comentarios.")
            continue

        modelo = IsolationForest(
            n_estimators=A.arboles,
            contamination=A.contaminacion,
            random_state=A.semilla,
            n_jobs=-1
        )
        pred = modelo.fit_predict(Z)
        scores = -modelo.score_samples(Z)
        df.loc[indices, "outlier"] = pred == -1
        df.loc[indices, "if_score"] = scores

        # Término más representativo de la fila estandarizada, como pista descriptiva.
        for local_i, global_i in enumerate(indices):
            fila = Z.getrow(local_i)
            if fila.nnz:
                j = fila.indices[np.argmax(fila.data)]
                df.loc[global_i, "if_motivo"] = nombres[j]

        n_out = int((pred == -1).sum())
        resumen.append({
            "fuente": fuente, "comentarios": len(indices), "atípicos": n_out,
            "% atípicos": round(100 * n_out / len(indices), 1), "nota": ""
        })
        print(f"\n{fuente}: {n_out} atípicos de {len(indices)} comentarios.")

    resumen = pd.DataFrame(resumen).set_index("fuente")
    titulo("Resumen de detección por base/fuente", nivel=2)
    mostrar_tabla(resumen)

    # Porcentaje de atípicos por fuente.
    fig, ax = plt.subplots(figsize=(max(9, len(resumen) * 1.15), 5.5))
    colores = [C_DMO if str(f).strip().lower() == str(A.dmo).strip().lower()
               else C_PERC for f in resumen.index]
    barras = ax.bar(resumen.index.astype(str), resumen["% atípicos"],
                    color=colores, edgecolor="black", hatch="///")
    for b, v in zip(barras, resumen["% atípicos"]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Porcentaje de comentarios atípicos por fuente — Isolation Forest")
    ax.set_ylabel("Comentarios atípicos (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()

    # GRÁFICAS GENERALES 2D Y 3D: concatenar todas las fuentes ya estandarizadas.
    # Los marcadores rojos son atípicos según el IF aplicado dentro de cada fuente.
    Z_global = vstack([matrices[f][1] for f in matrices]).tocsr()
    indices_globales = np.concatenate([matrices[f][0] for f in matrices])
    flags_globales = df.loc[indices_globales, "outlier"].to_numpy(dtype=bool)
    _graficar_proyeccion_2d_3d(
        Z_global, flags_globales,
        "TODAS LAS BASES — normales vs atípicos", A.semilla
    )

    # Gráficas individuales 2D y 3D por fuente.
    for fuente, (indices, Z) in matrices.items():
        flags = df.loc[indices, "outlier"].to_numpy(dtype=bool)
        _graficar_proyeccion_2d_3d(
            Z, flags, f"Fuente: {fuente}", A.semilla
        )

    # Distribución de scores por fuente.
    fig, ax = plt.subplots(figsize=(max(9, len(resumen) * 1.15), 5.5))
    paleta = {f: (C_DMO if str(f).strip().lower() == str(A.dmo).strip().lower()
                  else OKABE[i % len(OKABE)])
              for i, f in enumerate(df["fuente"].drop_duplicates())}
    sns.boxplot(data=df, x="fuente", y="if_score", hue="fuente",
                palette=paleta, dodge=False, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Distribución del score de anomalía por fuente")
    ax.set_xlabel("Fuente")
    ax.set_ylabel("Score (mayor = más atípico)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    return df


def tabla_ngramas(textos, n, top=15):
    try:
        textos = [str(x) for x in textos if str(x).strip()]
        cv = CountVectorizer(ngram_range=(n, n), token_pattern=r"(?u)\b\w+\b",
                             lowercase=False)
        M = cv.fit_transform(textos)
        frec = np.asarray(M.sum(axis=0)).ravel()
        docs = np.asarray((M > 0).sum(axis=0)).ravel()
        out = pd.DataFrame({"termino": cv.get_feature_names_out(),
                            "frecuencia": frec, "comentarios": docs})
        return out.sort_values(["frecuencia", "comentarios"], ascending=False).head(top)
    except ValueError:
        return pd.DataFrame(columns=["termino", "frecuencia", "comentarios"])


def graficar_ngramas(df, por_fuente=True, top=15):
    grupos = [("Corpus normal", df)]
    if por_fuente:
        grupos += [(str(f), g) for f, g in df.groupby("fuente")]
    for nombre, grupo in grupos:
        if grupo.empty:
            continue
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for ax, n, titulo_n, color, hatch in zip(
            axs, (1, 2, 3), ("Unigramas", "Bigramas", "Trigramas"),
            (C_DMO, C_PERC, C_POS), ("///", "...", "\\\\\\\\")):
            tabla = tabla_ngramas(grupo["texto_limpio"], n, top)
            if tabla.empty:
                ax.set_title(titulo_n + " (sin datos)")
                continue
            tabla = tabla.sort_values("frecuencia")
            barras = ax.barh(tabla["termino"], tabla["frecuencia"], color=color,
                             edgecolor="black", hatch=hatch)
            for b, v in zip(barras, tabla["frecuencia"]):
                ax.text(v, b.get_y() + b.get_height()/2, f" {int(v)}",
                        va="center", fontsize=8)
            ax.set_title(titulo_n)
            ax.set_xlabel("Frecuencia")
        fig.suptitle(f"N-gramas — {nombre} (n={len(grupo)})")
        fig.tight_layout()


def mostrar_ngramas(df):
    normales = df.loc[~df["outlier"]]
    titulo("3. UNIGRAMAS, BIGRAMAS Y TRIGRAMAS DE COMENTARIOS NORMALES")
    for fuente, grupo in [("GLOBAL", normales)] + list(normales.groupby("fuente")):
        print(f"\n{fuente}: {len(grupo)} comentarios normales")
        for n, nombre in [(1, "UNIGRAMAS"), (2, "BIGRAMAS"), (3, "TRIGRAMAS")]:
            print(f"\n{nombre}")
            print(tabla_ngramas(grupo["texto_limpio"], n, 20).to_string(index=False))
    graficar_ngramas(normales)


def mostrar_atipicos_y_nubes(df, A):
    """Imprime todos los atípicos por fuente y crea nubes globales y por fuente."""
    titulo("4. COMENTARIOS ATÍPICOS POR BASE/FUENTE")
    atipicos = df.loc[df["outlier"]].copy()
    print(f"\nTOTAL GENERAL: {len(atipicos)} comentarios atípicos de {len(df)} registros "
          f"({100 * len(atipicos) / max(len(df), 1):.2f}%).")
    if atipicos.empty:
        print("No se detectaron comentarios atípicos.")
    else:
        print("\nLISTADO GENERAL DE ATÍPICOS (todas las fuentes):")
        columnas_globales = [c for c in ["fuente", "texto_original", "if_score", "if_motivo"]
                             if c in atipicos.columns]
        print(atipicos.sort_values("if_score", ascending=False)[columnas_globales]
              .to_string(index=False, max_colwidth=350))
        for fuente, grupo in df.groupby("fuente", sort=False):
            sub = grupo.loc[grupo["outlier"]].sort_values("if_score", ascending=False)
            titulo(f"ATÍPICOS — {fuente}", nivel=2)
            print(f"Total de comentarios: {len(grupo)} | Atípicos: {len(sub)} | "
                  f"Normales: {len(grupo) - len(sub)}")
            if sub.empty:
                print("No se detectaron atípicos en esta fuente.")
                continue
            columnas = [c for c in ["texto_original", "if_score", "if_motivo"] if c in sub.columns]
            print(sub[columnas].to_string(index=False, max_colwidth=350))

    titulo("5. NUBES DE PALABRAS DE LOS COMENTARIOS ATÍPICOS")
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("No está instalada wordcloud. Instálala con: pip install wordcloud")
        return

    grupos = [("GLOBAL", atipicos)]
    grupos += [(str(f), g.loc[g["outlier"]]) for f, g in df.groupby("fuente", sort=False)]
    for nombre, grupo in grupos:
        if grupo.empty:
            print(f"Sin comentarios atípicos para la nube: {nombre}")
            continue
        texto = " ".join(grupo["texto_limpio"].dropna().astype(str))
        if not texto.strip():
            print(f"Sin texto limpio para la nube: {nombre}")
            continue
        es_dmo = bool(grupo["es_dmo"].all()) if "es_dmo" in grupo else False
        color = C_DMO if es_dmo else (OKABE[3] if nombre == "GLOBAL" else C_PERC)
        nube = WordCloud(
            width=1200, height=650, background_color="white", max_words=100,
            collocations=False, prefer_horizontal=0.9, random_state=A.semilla,
            color_func=lambda *args, _color=color, **kwargs: _color
        ).generate(texto)
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.imshow(nube, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Nube de palabras — Atípicos: {nombre} (n={len(grupo)})",
                     fontsize=15, fontweight="bold")
        fig.tight_layout()


def ejecutar_base(A):
    """Carga archivos, conserva la fuente y prepara el texto para el análisis."""
    rutas = list(A.archivos)
    if not rutas:
        print("\nIntroduce las rutas de los archivos (una por línea).")
        print("Cuando termines, presiona Enter en una línea vacía.\n")
        while True:
            ruta = input("Ruta del archivo: ").strip().strip('"').strip("'")
            if not ruta:
                break
            rutas.append(ruta)
    if not rutas:
        raise SystemExit("No se proporcionaron archivos para analizar.")

    df_raw = cargar_fuentes(rutas, A.col_texto, A.col_fuente)
    fuentes = list(df_raw["fuente"].dropna().astype(str).unique())
    A.dmo = elegir_dmo(fuentes, A.dmo)
    df = preprocesar(df_raw, A)
    df["es_dmo"] = df["fuente"].astype(str).eq(str(A.dmo))

    print(f"\nFuente identificada como DMO: {A.dmo}")
    print("Las demás fuentes se consideran imagen percibida.")
    return df




# ============================================================
# SENTIMIENTO HÍBRIDO (BERT + LÉXICO)
# ============================================================
import html as _html_mod
from matplotlib.patches import Patch
from sklearn.metrics import f1_score
import math

FIGS = []          # (nombre_archivo, figura) para --guardar
POS, NEG = "positivo", "negativo"
C_POS, H_POS = C_POS, "\\\\"
C_NEG, H_NEG = C_NEG, "xx"
C_DMO, C_PERC = C_DMO, C_PERC
FIGS = []          # (nombre_archivo, figura) para --guardar
# Colores de las fuentes percibidas: evitan el verde y el bermellón (reservados a pos/neg)
C_FUENTES = [OKABE[5], OKABE[3], OKABE[6], OKABE[7], OKABE[1]]


# ============================================================
# 1. ENFOQUE BERT (Transformers)
# ============================================================

def _valores_clase(id2label):
    """Vector con la polaridad (-1..+1) de cada clase del modelo.
    Modelos de estrellas (1-5): (estrella-3)/2.  Modelos neg/neu/pos: -1/0/+1."""
    n = len(id2label)
    nombres = [str(id2label[i]).lower() for i in range(n)]
    if n == 5 and all(re.search(r"[1-5]", s) for s in nombres):
        return np.array([(int(re.search(r"[1-5]", s).group()) - 3) / 2 for s in nombres])
    vals = []
    for s in nombres:
        if "neg" in s:
            vals.append(-1.0)
        elif "pos" in s:
            vals.append(1.0)
        elif "neu" in s:
            vals.append(0.0)
        else:
            vals = None
            break
    if vals is not None:
        return np.array(vals)
    if n == 2:                                   # LABEL_0 / LABEL_1
        return np.array([-1.0, 1.0])
    if n == 3:                                   # LABEL_0 / 1 / 2
        return np.array([-1.0, 0.0, 1.0])
    raise ValueError(f"No sé interpretar las clases del modelo: {nombres}")


def _texto_para_bert(t):
    """Limpieza mínima: conserva oración, puntuación, negaciones y emojis."""
    t = html.unescape(str(t))
    t = RE_URL.sub(" ", t)
    t = RE_MENCION.sub(" ", t)
    return " ".join(t.split())


def polaridad_bert(textos, A):
    """Polaridad BERT en [-1, 1] por comentario, o None si no se puede usar BERT."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        print("\n*** AVISO: no están instalados torch/transformers -> no se puede usar BERT.\n"
              "    Instala con: pip install torch transformers\n"
              "    Se continúa SOLO con el enfoque de reglas (léxico). ***")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(A.modelo)
        mod = AutoModelForSequenceClassification.from_pretrained(A.modelo)
        valores = _valores_clase(mod.config.id2label)
    except Exception as e:
        print(f"\n*** AVISO: no pude cargar el modelo '{A.modelo}' ({type(e).__name__}: {e}).\n"
              "    Revisa tu conexión (la primera vez se descarga) o el nombre del modelo.\n"
              "    Se continúa SOLO con el enfoque de reglas (léxico). ***")
        return None

    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    mod.to(dev).eval()
    print(f"BERT: {A.modelo} | dispositivo: {dev} | clases: "
          f"{[mod.config.id2label[i] for i in range(len(valores))]}")

    textos = [_texto_para_bert(t) for t in textos]
    orden = np.argsort([len(t) for t in textos])       # lotes de longitud parecida = más rápido
    out = np.zeros(len(textos))
    total = len(orden)
    with torch.no_grad():
        for k in range(0, total, A.lote):
            idx = orden[k:k + A.lote]
            enc = tok([textos[i] for i in idx], padding=True, truncation=True,
                      max_length=A.max_len, return_tensors="pt").to(dev)
            prob = torch.softmax(mod(**enc).logits, dim=-1).cpu().numpy()
            out[idx] = prob @ valores
            hecho = min(k + A.lote, total)
            if (k // A.lote) % 10 == 0 or hecho == total:
                print(f"  BERT: {hecho:,}/{total:,} comentarios", end="\r")
    print()
    return out


# ============================================================
# 2. FUSIÓN HÍBRIDA Y CLASIFICACIÓN BINARIA
# ============================================================

def clasificar_hibrido(df, A):
    """Agrega: pol_lexico, pol_bert, polaridad, metodo, etiqueta, etiqueta_modelo."""
    df = df.copy()
    x = df["puntaje_bruto"].to_numpy(float)
    df["pol_lexico"] = np.tanh(x / 2.0)                 # acota el puntaje bruto a (-1, 1)
    df["tiene_lexico"] = x != 0                         # 0 = sin señal (o señales que se anulan)

    bert = None if A.sin_bert else polaridad_bert(df["texto_original"].tolist(), A)
    df["pol_bert"] = bert if bert is not None else np.nan
    pb, pl, tl = df["pol_bert"].to_numpy(float), df["pol_lexico"].to_numpy(float), \
        df["tiene_lexico"].to_numpy()

    if bert is not None:
        w = float(np.clip(A.peso_bert, 0, 1))
        pol = np.where(tl, w * pb + (1 - w) * pl, pb)
        df["metodo"] = np.where(tl, "BERT + léxico", "solo BERT")
    else:
        pol = np.where(tl, pl, np.nan)
        df["metodo"] = np.where(tl, "solo léxico", "sin señal")
    df["polaridad"] = pol
    df["confianza"] = np.abs(pol)

    et = np.where(np.isnan(pol), "sin señal", np.where(pol >= A.umbral, POS, NEG))
    df["etiqueta_modelo"] = et
    df["etiqueta"] = et
    if not A.dmo_puede_ser_negativa:
        df.loc[df["es_dmo"], "etiqueta"] = POS          # DMO positiva por diseño
    return df


def separar_particiones(df):
    """Dos grupos (positivos / negativos) con la etiqueta de fuente conservada."""
    pos = df.loc[df["etiqueta"] == POS].copy()
    neg = df.loc[df["etiqueta"] == NEG].copy()
    return pos, neg


# ============================================================
# 3. REPORTE EN CONSOLA
# ============================================================

COLS_SALIDA = ["fuente", "idioma", "texto_original", "polaridad", "pol_bert",
               "pol_lexico", "metodo", "etiqueta"]


def tabla_por_fuente(df, A):
    t = (df.groupby(["fuente", "etiqueta"]).size().unstack(fill_value=0)
           .reindex(columns=[POS, NEG], fill_value=0))
    t["total"] = t.sum(axis=1)
    t["% positivo"] = (100 * t[POS] / t["total"]).round(1)
    t["% negativo"] = (100 * t[NEG] / t["total"]).round(1)
    t["polaridad media"] = df.groupby("fuente")["polaridad"].mean().round(3)
    t["es_dmo"] = [str(f) == str(A.dmo) for f in t.index]
    return t.sort_values(["es_dmo", "% negativo"], ascending=[False, False])


def etiquetas_estrellas(df):
    """4-5 estrellas = positivo, 1-2 = negativo, 3 = se ignora. (calif_norm = estrellas/5)"""
    est = df["calif_norm"] * 5
    return pd.Series(np.where(est >= 4, POS, np.where(est <= 2, NEG, None)),
                     index=df.index).where(est.notna())


def validar_con_estrellas(df):
    """Compara BERT, léxico e híbrido contra las estrellas (solo fuentes percibidas)."""
    ref = etiquetas_estrellas(df)
    ok = ref.notna() & ~df["es_dmo"]
    if ok.sum() < 5:
        return None
    sub = df.loc[ok].copy()
    sub["ref"] = ref[ok]
    metodos = {"Híbrido": sub["etiqueta_modelo"]}
    if sub["pol_bert"].notna().any():
        metodos["Solo BERT"] = pd.Series(np.where(sub["pol_bert"] >= 0, POS, NEG), index=sub.index)
    metodos["Solo léxico"] = pd.Series(
        np.where(sub["pol_lexico"] > 0, POS, np.where(sub["pol_lexico"] < 0, NEG, None)),
        index=sub.index)
    filas = []
    for nombre, pred in metodos.items():
        m = pred.notna() & pred.ne("sin señal")
        if m.sum() == 0:
            continue
        y, p = sub.loc[m, "ref"], pred[m]
        filas.append({"método": nombre, "comentarios evaluados": int(m.sum()),
                      "cobertura %": round(100 * m.mean(), 1),
                      "exactitud": round(float((y == p).mean()), 3),
                      "F1 macro": round(float(f1_score(y, p, average="macro")), 3)})
    return pd.DataFrame(filas).set_index("método"), sub


def imprimir_reporte(normales, pos, neg, A, n_total, n_out):
    titulo("RESULTADO DEL ANÁLISIS DE SENTIMIENTOS (solo comentarios normales)")
    print(f"Comentarios totales tras limpieza : {n_total:,}")
    print(f"Atípicos excluidos                : {n_out:,}")
    print(f"Comentarios normales analizados   : {len(normales):,}")
    print(f"Método                            : "
          + ", ".join(f"{k}={v}" for k, v in normales['metodo'].value_counts().items()))
    print(f"Umbral de polaridad               : {A.umbral:+.2f}  "
          f"(>= umbral = positivo, < umbral = negativo)")
    if not A.sin_bert and normales["pol_bert"].notna().any():
        print(f"Pesos de la fusión                : BERT {A.peso_bert:.2f} | léxico {1 - A.peso_bert:.2f}")
    print(f"\nGRUPO POSITIVO: {len(pos):,} comentarios   |   GRUPO NEGATIVO: {len(neg):,} comentarios")
    sin = int((normales["etiqueta"] == "sin señal").sum())
    if sin:
        print(f"Sin señal (excluidos de ambos grupos): {sin:,}")

    titulo("Distribución por fuente", nivel=2)
    mostrar_tabla(tabla_por_fuente(normales, A).drop(columns="es_dmo"))

    dmo = normales[normales["es_dmo"]]
    if len(dmo) and not A.dmo_puede_ser_negativa:
        n_neg_modelo = int((dmo["etiqueta_modelo"] == NEG).sum())
        print(f"\nDMO ({A.dmo}): etiquetada POSITIVA por diseño. El modelo habría marcado "
              f"{n_neg_modelo} de {len(dmo)} como negativos ({100 * n_neg_modelo / len(dmo):.1f}%).")

    for nombre, grupo, asc in (("POSITIVOS", pos, False), ("NEGATIVOS", neg, True)):
        titulo(f"Comentarios más {nombre.lower()} (mayor |polaridad|)", nivel=2)
        if grupo.empty:
            print("(sin comentarios)")
            continue
        top = grupo.sort_values("polaridad", ascending=asc).head(8).copy()
        top["texto_original"] = top["texto_original"].map(lambda s: recortar(s, 130))
        mostrar_tabla(top[["fuente", "polaridad", "texto_original"]].round(3), index=False,
                        justify="left")

    if normales["pol_bert"].notna().any() and normales["tiene_lexico"].any():
        c = normales[normales["tiene_lexico"]]
        acuerdo = ((c["pol_bert"] >= 0) == (c["pol_lexico"] > 0)).mean()
        print(f"\nConcordancia BERT vs léxico (comentarios con señal léxica, n={len(c):,}): "
              f"{100 * acuerdo:.1f}%")

    val = validar_con_estrellas(normales)
    if val is not None:
        titulo("Validación contra estrellas (4-5 = positivo, 1-2 = negativo; sin la DMO)", nivel=2)
        mostrar_tabla(val[0])
    else:
        print("\n(No hay estrellas suficientes en las fuentes percibidas para validar.)")

    titulo("Tamaño de las particiones para el paso 4 (tópicos vs frecuencias)", nivel=2)
    for nombre, grupo in (("positiva", pos), ("negativa", neg)):
        enfoque = "frecuencia de palabras" if len(grupo) < A.umbral_particion else "modelado de tópicos"
        print(f"Partición {nombre}: {len(grupo):,} comentarios | umbral {A.umbral_particion} "
              f"-> {enfoque}")
        print("   por fuente: " + (", ".join(f"{k}={v}" for k, v in
                                            grupo['fuente'].value_counts().items()) or "—"))


# ============================================================
# 4. GRÁFICAS (Okabe-Ito; color + trama/forma; etiquetas numéricas directas)
# ============================================================

def _reg(nombre, fig):
    fig.tight_layout()
    FIGS.append((nombre, fig))


def g_distribucion_fuentes(normales, A, metodo):
    t = tabla_por_fuente(normales, A)
    fuentes = [str(f) for f in t.index][::-1]
    t = t.iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(15, max(4.5, 0.8 * len(t) + 2.5)))

    ax[0].barh(y, t[POS], color=C_POS, edgecolor="black", hatch=H_POS, label="Positivo")
    ax[0].barh(y, t[NEG], left=t[POS], color=C_NEG, edgecolor="black", hatch=H_NEG,
               label="Negativo")
    for i, (p, n) in enumerate(zip(t[POS], t[NEG])):
        if p:
            ax[0].text(p / 2, i, f"{int(p)}", ha="center", va="center", fontsize=9,
                       bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
        if n:
            ax[0].text(p + n / 2, i, f"{int(n)}", ha="center", va="center", fontsize=9,
                       bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax[0].set_title("Comentarios por grupo y fuente")
    ax[0].set_xlabel("Comentarios normales")

    ax[1].barh(y, t["% positivo"], color=C_POS, edgecolor="black", hatch=H_POS)
    ax[1].barh(y, t["% negativo"], left=t["% positivo"], color=C_NEG, edgecolor="black",
               hatch=H_NEG)
    for i, (p, n) in enumerate(zip(t["% positivo"], t["% negativo"])):
        if p >= 4:
            ax[1].text(p / 2, i, f"{p:.0f}%", ha="center", va="center", fontsize=9,
                       bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
        if n >= 4:
            ax[1].text(p + n / 2, i, f"{n:.0f}%", ha="center", va="center", fontsize=9,
                       bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax[1].set_xlim(0, 100)
    ax[1].set_title("Proporción positivo / negativo (100 %)")
    ax[1].set_xlabel("% de comentarios")
    for a in ax:
        a.set_yticks(y)
        a.set_yticklabels([f + (" (DMO)" if f == str(A.dmo) else "") for f in fuentes])
    ax[0].legend(handles=[Patch(fc=C_POS, ec="black", hatch=H_POS, label="Positivo"),
                          Patch(fc=C_NEG, ec="black", hatch=H_NEG, label="Negativo")],
                 loc="lower right")
    fig.suptitle(f"Sentimiento por fuente — comentarios normales ({metodo})",
                 fontsize=14, fontweight="bold")
    _reg("01_sentimiento_por_fuente", fig)


def g_polaridad(normales, A, metodo):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    p = normales["polaridad"].dropna()
    bins = np.linspace(-1, 1, 41)
    ax[0].hist(p[p < A.umbral], bins=bins, color=C_NEG, edgecolor="black", hatch=H_NEG,
               label=f"Negativo (n={int((p < A.umbral).sum())})")
    ax[0].hist(p[p >= A.umbral], bins=bins, color=C_POS, edgecolor="black", hatch=H_POS,
               label=f"Positivo (n={int((p >= A.umbral).sum())})")
    ax[0].axvline(A.umbral, color="black", ls="--", lw=1.5)
    ax[0].set_title("Distribución de la polaridad híbrida")
    ax[0].set_xlabel("Polaridad (−1 muy negativo … +1 muy positivo)")
    ax[0].set_ylabel("Comentarios")
    ax[0].legend()
    ax[0].grid(alpha=.2)

    fuentes = [str(f) for f in normales["fuente"].drop_duplicates()]
    datos = [normales.loc[normales["fuente"].astype(str) == f, "polaridad"].dropna() for f in fuentes]
    bp = ax[1].boxplot(datos, vert=False, patch_artist=True, showmeans=True,
                       meanprops=dict(marker="D", mfc="white", mec="black", ms=7))
    for i, (caja, f) in enumerate(zip(bp["boxes"], fuentes)):
        es_dmo = f == str(A.dmo)
        caja.set(facecolor=C_DMO if es_dmo else C_FUENTES[i % len(C_FUENTES)],
                 hatch="///" if es_dmo else ["...", "\\\\", "xx", "++", "oo"][i % 5])
    for i, d in enumerate(datos, 1):
        if len(d):
            ax[1].text(1.02, i, f"n={len(d)}  μ={d.mean():+.2f}", va="center", fontsize=9,
                       transform=ax[1].get_yaxis_transform())
    ax[1].axvline(A.umbral, color="black", ls="--", lw=1.5)
    ax[1].set_yticklabels([f + (" (DMO)" if f == str(A.dmo) else "") for f in fuentes])
    ax[1].set_xlim(-1.05, 1.05)
    ax[1].set_xlabel("Polaridad (◆ = media)")
    ax[1].set_title("Polaridad por fuente")
    ax[1].grid(axis="x", alpha=.2)
    fig.suptitle(f"Polaridad y umbral de clasificación ({metodo})", fontsize=14, fontweight="bold")
    _reg("02_polaridad", fig)


def g_bert_vs_lexico(normales, A):
    d = normales[normales["tiene_lexico"] & normales["pol_bert"].notna()]
    if d.empty:
        return
    rng = np.random.default_rng(A.semilla)
    jit = rng.normal(0, 0.025, len(d))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for et, color, marca, tam in ((POS, C_POS, "o", 26), (NEG, C_NEG, "X", 48)):
        m = (d["etiqueta_modelo"] == et).to_numpy()
        ax.scatter(d["pol_lexico"].to_numpy()[m] + jit[m], d["pol_bert"].to_numpy()[m],
                   c=color, marker=marca, s=tam, alpha=.6, edgecolors="black", linewidths=.4,
                   label=f"{et.capitalize()} (híbrido) n={int(m.sum())}")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    acuerdo = ((d["pol_bert"] >= 0) == (d["pol_lexico"] > 0)).mean()
    ax.set_title(f"BERT vs léxico — concordancia {100 * acuerdo:.1f}% (n={len(d)})")
    ax.set_xlabel("Polaridad del léxico (con jitter)")
    ax.set_ylabel("Polaridad de BERT")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=.2)
    ax.legend(loc="lower right")
    _reg("03_bert_vs_lexico", fig)


def g_validacion(normales):
    val = validar_con_estrellas(normales)
    if val is None:
        return
    met, sub = val
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    cm = pd.crosstab(sub["ref"], sub["etiqueta_modelo"]).reindex(index=[POS, NEG],
                                                                columns=[POS, NEG], fill_value=0)
    im = ax[0].imshow(cm.values, cmap=CMAP, vmin=0, vmax=max(cm.values.max(), 1))
    for i in range(2):
        for j in range(2):
            v = cm.values[i, j]
            ax[0].text(j, i, f"{v}\n({100 * v / max(cm.values[i].sum(), 1):.0f}% de la fila)",
                       ha="center", va="center", fontsize=11,
                       color=_color_texto(v / max(cm.values.max(), 1)))
    ax[0].set_xticks([0, 1], [f"Pred. {POS}", f"Pred. {NEG}"])
    ax[0].set_yticks([0, 1], [f"Estrellas: {POS}", f"Estrellas: {NEG}"])
    ax[0].set_title("Matriz de confusión: híbrido vs estrellas")
    fig.colorbar(im, ax=ax[0], fraction=.046)

    x = np.arange(len(met))
    ancho = .38
    b1 = ax[1].bar(x - ancho / 2, met["exactitud"], ancho, color=C_DMO, edgecolor="black",
                   hatch="///", label="Exactitud")
    b2 = ax[1].bar(x + ancho / 2, met["F1 macro"], ancho, color=C_PERC, edgecolor="black",
                   hatch="...", label="F1 macro")
    for b in list(b1) + list(b2):
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                   ha="center", va="bottom", fontsize=10)
    ax[1].set_xticks(x, [f"{m}\n(n={int(n)}, cob. {c:.0f}%)" for m, n, c in
                         zip(met.index, met["comentarios evaluados"], met["cobertura %"])])
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title("Calidad frente a las estrellas")
    ax[1].legend(loc="upper center", ncol=2)
    ax[1].grid(axis="y", alpha=.25)
    fig.suptitle("Validación del clasificador (fuentes percibidas con calificación)",
                 fontsize=14, fontweight="bold")
    _reg("04_validacion_estrellas", fig)


def g_terminos(pos, neg):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for col, (nombre, grupo, color, hatch) in enumerate((("Positivos", pos, C_POS, H_POS),
                                                          ("Negativos", neg, C_NEG, H_NEG))):
        for fila, (n, nom_n) in enumerate(((1, "Unigramas"), (2, "Bigramas"))):
            ax = axs[fila, col]
            tabla = tabla_ngramas(grupo["texto_limpio"], n, 15) if len(grupo) else pd.DataFrame()
            if tabla.empty:
                ax.set_title(f"{nom_n} — {nombre} (sin datos)")
                ax.axis("off")
                continue
            tabla = tabla.sort_values("frecuencia")
            barras = ax.barh(tabla["termino"], tabla["frecuencia"], color=color,
                             edgecolor="black", hatch=hatch)
            for b, v in zip(barras, tabla["frecuencia"]):
                ax.text(v, b.get_y() + b.get_height() / 2, f" {int(v)}", va="center", fontsize=8)
            ax.set_title(f"{nom_n} — {nombre} (n={len(grupo)})")
            ax.set_xlabel("Frecuencia")
    fig.suptitle("Términos más frecuentes por grupo de sentimiento", fontsize=14,
                 fontweight="bold")
    _reg("05_terminos_por_grupo", fig)


def g_nubes(pos, neg, A):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("\n(Nubes omitidas: instala wordcloud con  pip install wordcloud)")
        return
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, nombre, grupo, color in ((axs[0], "Positivos", pos, C_POS),
                                     (axs[1], "Negativos", neg, C_NEG)):
        texto = " ".join(grupo["texto_limpio"].dropna().astype(str))
        ax.axis("off")
        if not texto.strip():
            ax.set_title(f"{nombre} (sin texto)")
            continue
        nube = WordCloud(width=1000, height=600, background_color="white", max_words=100,
                         collocations=False, random_state=A.semilla,
                         color_func=lambda *a, _c=color, **k: _c).generate(texto)
        ax.imshow(nube, interpolation="bilinear")
        ax.set_title(f"{nombre} (n={len(grupo)})", fontsize=14, fontweight="bold")
    _reg("06_nubes_por_grupo", fig)


def g_fuente_por_grupo(pos, neg, A):
    """¿De qué fuentes viene cada grupo? (composición de cada partición)"""
    fuentes = [str(f) for f in pd.concat([pos["fuente"], neg["fuente"]]).drop_duplicates()]
    if not fuentes:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for a, nombre, grupo, hatch in ((ax[0], "Positivos", pos, H_POS), (ax[1], "Negativos", neg, H_NEG)):
        c = grupo["fuente"].astype(str).value_counts().reindex(fuentes, fill_value=0)
        colores = [C_DMO if f == str(A.dmo) else C_FUENTES[i % len(C_FUENTES)]
                   for i, f in enumerate(fuentes)]
        barras = a.bar(range(len(c)), c.values, color=colores, edgecolor="black", hatch=hatch)
        tot = max(c.sum(), 1)
        for b, v in zip(barras, c.values):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v}\n{100 * v / tot:.0f}%", ha="center",
                   va="bottom", fontsize=9)
        a.set_xticks(range(len(c)), c.index, rotation=30, ha="right")
        a.set_title(f"Grupo {nombre.lower()}: composición por fuente (n={int(c.sum())})")
        a.set_ylabel("Comentarios")
        a.set_ylim(0, max(c.max() * 1.2, 1))
        a.grid(axis="y", alpha=.25)
    _reg("07_composicion_grupos", fig)


# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================



def crear_parser():
    p = argparse.ArgumentParser(description="Sentimiento híbrido BERT + léxico sobre "
                                            "comentarios normales (tras Isolation Forest)")
    p.add_argument("archivos", nargs="*")
    # --- mismos parámetros que limpieza_outliers.py ---
    p.add_argument("--dmo", default=None)
    p.add_argument("--col-texto", default=None)
    p.add_argument("--col-fuente", default=None)
    p.add_argument("--idioma", default="auto")
    p.add_argument("--max-por-fuente", type=int, default=20000)
    p.add_argument("--arboles", type=int, default=300)
    p.add_argument("--contaminacion", type=float, default=.05)
    p.add_argument("--min-if", type=int, default=20)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--sin-graficas", action="store_true")
    # --- sentimiento ---
    p.add_argument("--modelo", default="nlptown/bert-base-multilingual-uncased-sentiment",
                   help="modelo de Hugging Face para sentimiento")
    p.add_argument("--sin-bert", action="store_true", help="usar solo el léxico")
    p.add_argument("--peso-bert", type=float, default=0.6,
                   help="peso de BERT en la fusión (0-1); el léxico pesa 1 - peso")
    p.add_argument("--umbral", type=float, default=0.0,
                   help="umbral de polaridad: >= umbral positivo, < umbral negativo")
    p.add_argument("--max-len", type=int, default=256, help="tokens máximos por comentario en BERT")
    p.add_argument("--lote", type=int, default=32, help="tamaño de lote en BERT")
    p.add_argument("--umbral-particion", type=int, default=30)
    p.add_argument("--dmo-puede-ser-negativa", action="store_true")
    p.add_argument("--ver-graficas-outliers", action="store_true",
                   help="no cerrar las gráficas del paso de atípicos")
    p.add_argument("--guardar", default=None, metavar="CARPETA",
                   help="OPCIONAL: guarda PNG y CSV de las particiones (por omisión no se guarda nada)")
    return p






# ============================================================
# TÓPICOS POR SENTIMIENTO
# ============================================================
HATCHES = ["...", "\\\\", "xx", "++", "oo", "--"]


# ============================================================
# 1. DECISIÓN DEL ENFOQUE (condición especial)
# ============================================================

def decidir_enfoque(n, A):
    """Devuelve (enfoque, motivo) según el tamaño de la partición."""
    if n == 0:
        return "ninguno", "la partición está vacía"
    if n < A.umbral_particion:
        return "frecuencias", (f"{n} comentarios < umbral {A.umbral_particion}: demasiado "
                               "pequeña para estimar tópicos estables")
    return "tópicos", f"{n} comentarios >= umbral {A.umbral_particion}"


def anunciar(nombre, n, enfoque, motivo, A):
    titulo(f"PARTICIÓN {nombre.upper()}", nivel=2)
    print(f"  Tamaño   : {n:,} comentarios normales")
    print(f"  Umbral   : {A.umbral_particion}")
    print(f"  Enfoque  : {'MODELADO DE TÓPICOS' if enfoque == 'tópicos' else 'FRECUENCIA DE PALABRAS' if enfoque == 'frecuencias' else 'ninguno'}")
    print(f"  Motivo   : {motivo}")


# ============================================================
# 2. MODELADO DE TÓPICOS
# ============================================================

def vectorizar(textos):
    n = len(textos)
    kw = dict(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if n >= 60 else 1, max_df=0.8, **kw)
        X = vec.fit_transform(textos)
    except ValueError:                        # vocabulario vacío por los filtros
        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1, **kw)
        X = vec.fit_transform(textos)
    return vec, X.tocsr()


def coherencia_npmi(Xb, tops, eps=1e-12):
    """Coherencia NPMI media de los tópicos (co-ocurrencia en comentarios). Más alto = mejor."""
    cols = sorted({j for t in tops for j in t})
    pos = {j: i for i, j in enumerate(cols)}
    M = Xb[:, cols].toarray().astype(float)
    co, p = (M.T @ M) / M.shape[0], M.mean(axis=0)
    vals = []
    for t in tops:
        s = []
        for a in range(len(t)):
            for b in range(a + 1, len(t)):
                i, j = pos[t[a]], pos[t[b]]
                pij = co[i, j]
                s.append(np.log((pij + eps) / (p[i] * p[j] + eps)) / (-np.log(pij + eps))
                         if pij > 0 else -1.0)
        vals.append(np.mean(s) if s else 0.0)
    return float(np.mean(vals))


def modelar_nmf(textos, A, n_terminos=10):
    vec, X = vectorizar(textos)
    if X.shape[1] < 3:
        return None
    terms = vec.get_feature_names_out()
    Xb = (X > 0).astype(float).tocsr()
    n = X.shape[0]
    if A.n_topicos:
        ks = [A.n_topicos]
    else:
        kmin = max(2, A.min_topicos)
        ks = list(range(kmin, max(kmin, min(A.max_topicos, n // 8)) + 1))
    mejor, curva = None, []
    for k in ks:
        k = min(k, X.shape[1] - 1)
        nmf = NMF(n_components=k, init="nndsvda", random_state=A.semilla, max_iter=600)
        W = nmf.fit_transform(X)
        H = nmf.components_
        tops = [np.argsort(H[t])[::-1][:n_terminos] for t in range(k)]
        coh = coherencia_npmi(Xb, tops)
        curva.append((k, coh))
        if mejor is None or coh > mejor[0] + 1e-9:
            mejor = (coh, k, W, H, tops)
    coh, k, W, H, tops = mejor
    asign = np.where(W.max(axis=1) > 0, W.argmax(axis=1), -1)
    return dict(metodo="NMF sobre TF-IDF (unigramas + bigramas)", k=k, coherencia=coh,
                curva=curva, W=W, X=X, asign=asign,
                terminos=[[(terms[j], float(H[t, j])) for j in tops[t]] for t in range(k)])


def modelar_bertopic(sub, A):
    """BERTopic opcional. Devuelve None (y avisa) si no se puede usar."""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception:
        print("  AVISO: BERTopic no está disponible (pip install bertopic sentence-transformers). "
              "Se usa NMF.")
        return None
    try:
        textos_emb = [_texto_para_bert(t) for t in sub["texto_original"]]
        emb = SentenceTransformer(A.bt_modelo).encode(textos_emb, show_progress_bar=False)
        vm = CountVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2))
        bt = BERTopic(vectorizer_model=vm, min_topic_size=max(5, A.bt_min_tam),
                      nr_topics=A.n_topicos or None, calculate_probabilities=False, verbose=False)
        topics, _ = bt.fit_transform(sub["texto_limpio"].tolist(), embeddings=emb)
        ids = sorted(t for t in set(topics) if t != -1)
        if not ids:
            print("  AVISO: BERTopic no encontró tópicos. Se usa NMF.")
            return None
        remap = {t: i for i, t in enumerate(ids)}
        asign = np.array([remap.get(t, -1) for t in topics])
        terminos = [[(w, float(s)) for w, s in bt.get_topic(t)[:10]] for t in ids]
        _, X = vectorizar(sub["texto_limpio"].tolist())
        W = np.zeros((X.shape[0], len(ids)))
        for t in range(len(ids)):                      # similitud con el centroide del tópico
            idx = np.where(asign == t)[0]
            cen = np.asarray(X[idx].mean(axis=0)).ravel()
            nr = np.linalg.norm(cen)
            if nr > 0:
                W[idx, t] = np.asarray(X[idx] @ (cen / nr)).ravel()
        return dict(metodo=f"BERTopic ({A.bt_modelo})", k=len(ids), coherencia=np.nan, curva=[],
                    W=W, X=X, asign=asign, terminos=terminos)
    except Exception as e:
        print(f"  AVISO: BERTopic falló ({type(e).__name__}: {e}). Se usa NMF.")
        return None


def modelar(sub, A):
    res = modelar_bertopic(sub, A) if A.bertopic else None
    return res if res is not None else modelar_nmf(sub["texto_limpio"].tolist(), A)


# ============================================================
# 3. REPORTE EN CONSOLA
# ============================================================

def reportar_topicos(sub, res, nombre, A, nota_gap=False):
    k = res["k"]
    print(f"\n  Método   : {res['metodo']}")
    if res["curva"]:
        print("  Coherencia NPMI por k: " + ", ".join(f"k={a}: {b:.3f}" for a, b in res["curva"]))
    coh = f" | coherencia NPMI = {res['coherencia']:.3f}" if not np.isnan(res["coherencia"]) else ""
    print(f"  Tópicos  : {k}{coh}")
    asign = res["asign"]
    n = len(sub)
    sin = int((asign == -1).sum())
    for t in range(k):
        m = asign == t
        nt = int(m.sum())
        print(f"\n  TÓPICO {t + 1}  —  {nt} comentarios ({100 * nt / n:.1f}%)")
        print("    Términos : " + ", ".join(w for w, _ in res["terminos"][t][:10]))
        por_fuente = sub.loc[m, "fuente"].astype(str).value_counts()
        print("    Fuentes  : " + (", ".join(f"{f}={c}" for f, c in por_fuente.items()) or "—"))
        if res["W"] is not None and nt:
            idx = np.where(m)[0]
            top = idx[np.argsort(res["W"][idx, t])[::-1][:3]]
            for i in top:
                print(f"      · [{sub.iloc[i]['fuente']}] {recortar(sub.iloc[i]['texto_original'], 150)}")
    if sin:
        print(f"\n  Sin tópico asignado: {sin} comentarios ({100 * sin / n:.1f}%)")
    if nota_gap:
        print("\n  Lectura: la DMO está etiquetada positiva por diseño, así que cada tópico de esta "
              "partición\n  es una BRECHA: lo mencionan los usuarios y la DMO no lo comunica.")


def tabla_terminos(textos, n, top=15):
    return tabla_ngramas([t for t in textos if str(t).strip()], n, top)


def reportar_frecuencias(sub, A, top=15):
    for n, nom in ((1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")):
        t = tabla_terminos(sub["texto_limpio"], n, top)
        print(f"\n  {nom} más frecuentes:")
        if t.empty:
            print("    (sin términos)")
            continue
        vmax = t["frecuencia"].max()
        for _, r in t.iterrows():
            print(f"    {r['termino']:<32} {int(r['frecuencia']):>5}  "
                  f"({int(r['comentarios'])} coment.)  {barra(r['frecuencia'], vmax, 18)}")
    print("\n  Comentarios por fuente: " + ", ".join(
        f"{f}={c}" for f, c in sub["fuente"].astype(str).value_counts().items()))


# ============================================================
# 4. GRÁFICAS
# ============================================================

def _color_fuente(f, A, i):
    return C_DMO if str(f) == str(A.dmo) else C_FUENTES[i % len(C_FUENTES)]


def g_decision(pos, neg, A):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    datos = [("Positiva", len(pos), C_POS, H_POS), ("Negativa", len(neg), C_NEG, H_NEG)]
    barras = ax.bar([d[0] for d in datos], [d[1] for d in datos],
                    color=[d[2] for d in datos], edgecolor="black",
                    hatch=None)
    for b, d in zip(barras, datos):
        b.set_hatch(d[3])
        enf, _ = decidir_enfoque(d[1], A)
        ax.text(b.get_x() + b.get_width() / 2, d[1], f"{d[1]}\n{'TÓPICOS' if enf == 'tópicos' else 'FRECUENCIAS' if enf == 'frecuencias' else 'vacía'}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(A.umbral_particion, color="black", ls="--", lw=1.6)
    ax.text(1.5, A.umbral_particion, f" umbral = {A.umbral_particion}", va="bottom", ha="right")
    ax.set_ylim(0, max(len(pos), len(neg), A.umbral_particion) * 1.25 + 1)
    ax.set_ylabel("Comentarios normales")
    ax.set_title("Tamaño de cada partición frente al umbral\n(por debajo: frecuencia de palabras; "
                 "por encima: modelado de tópicos)")
    ax.grid(axis="y", alpha=.25)
    _reg("00_decision_enfoque", fig)


def g_coherencia(res, nombre, pref):
    if len(res["curva"]) < 2:
        return
    ks, cs = zip(*res["curva"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ks, cs, "-o", color=OKABE[0], lw=2, ms=8)
    ax.plot([res["k"]], [res["coherencia"]], marker="*", ms=22, color=OKABE[4],
            mec="black", ls="none", label=f"k elegido = {res['k']}")
    for a, b in zip(ks, cs):
        ax.annotate(f"{b:.3f}", (a, b), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=9)
    ax.set_xticks(list(ks))
    ax.set_xlabel("Número de tópicos (k)")
    ax.set_ylabel("Coherencia NPMI")
    ax.set_title(f"Selección del número de tópicos — {nombre}")
    ax.legend()
    ax.grid(alpha=.25)
    _reg(f"{pref}_1_coherencia", fig)


def g_terminos_topicos(res, nombre, color, hatch, pref):
    k = res["k"]
    ncol = min(k, 3)
    nrow = math.ceil(k / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.7 * nrow), squeeze=False)
    for t in range(nrow * ncol):
        ax = axs[t // ncol, t % ncol]
        if t >= k:
            ax.axis("off")
            continue
        terms = res["terminos"][t][:8][::-1]
        vals = [v for _, v in terms]
        barras = ax.barh([w for w, _ in terms], vals, color=color, edgecolor="black", hatch=hatch)
        for b, v in zip(barras, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}", va="center", fontsize=8)
        ax.set_title(f"Tópico {t + 1}  (n={int((res['asign'] == t).sum())})", fontweight="bold")
        ax.set_xlabel("Peso del término")
    fig.suptitle(f"Términos principales por tópico — {nombre}", fontsize=14, fontweight="bold")
    _reg(f"{pref}_2_terminos", fig)


def g_prevalencia_fuentes(sub, res, nombre, A, pref):
    k = res["k"]
    fuentes = [str(f) for f in sub["fuente"].drop_duplicates()]
    ct = pd.crosstab(sub["fuente"].astype(str), res["asign"]).reindex(
        index=fuentes, columns=list(range(k)), fill_value=0)
    fig, ax = plt.subplots(1, 2, figsize=(15, max(4.8, 0.7 * k + 2.8)),
                           gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(k)
    izq = np.zeros(k)
    percibidas = [f for f in fuentes if f != str(A.dmo)]
    for i, f in enumerate(fuentes):
        v = ct.loc[f].to_numpy(float)
        ax[0].barh(y, v, left=izq, color=_color_fuente(f, A, percibidas.index(f) if f in percibidas else 0),
                   edgecolor="black", hatch=HATCHES[i % len(HATCHES)], label=f)
        izq += v
    n = max(len(sub), 1)
    for t in range(k):
        ax[0].text(izq[t], t, f"  {int(izq[t])} ({100 * izq[t] / n:.0f}%)", va="center", fontsize=9)
    ax[0].set_yticks(y, [f"Tópico {t + 1}" for t in range(k)])
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, max(izq.max() * 1.25, 1))
    ax[0].set_xlabel("Comentarios")
    ax[0].set_title("Prevalencia de cada tópico por fuente")
    ax[0].legend(loc="upper right", fontsize=9)

    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    im = ax[1].imshow(pct.to_numpy(), cmap=CMAP, vmin=0, aspect="auto")
    for i in range(len(fuentes)):
        for j in range(k):
            v = pct.iloc[i, j]
            ax[1].text(j, i, f"{v:.0f}%\n(n={int(ct.iloc[i, j])})", ha="center", va="center",
                       fontsize=9, color=_color_texto(v / max(pct.to_numpy().max(), 1e-9)))
    ax[1].set_xticks(range(k), [f"T{t + 1}" for t in range(k)])
    ax[1].set_yticks(range(len(fuentes)),
                     [f + (" (DMO)" if f == str(A.dmo) else "") for f in fuentes])
    ax[1].set_title("% de los comentarios de cada fuente en cada tópico")
    fig.colorbar(im, ax=ax[1], fraction=.046)
    fig.suptitle(f"Tópicos y fuentes — {nombre}", fontsize=14, fontweight="bold")
    _reg(f"{pref}_3_topicos_fuentes", fig)


def g_mapa(res, nombre, A, pref):
    X = res["X"]
    if X.shape[0] < 3 or X.shape[1] < 2:
        return
    Z = TruncatedSVD(2, random_state=A.semilla).fit_transform(X)
    Z = Z + np.random.default_rng(A.semilla).normal(0, 0.012 * (Z.std(axis=0) + 1e-9), Z.shape)  # separa duplicados
    asign = res["asign"]
    fig, ax = plt.subplots(figsize=(10, 7.2))
    marcas = MARCAS
    for t in range(res["k"]):
        m = asign == t
        if not m.any():
            continue
        c = OKABE[t % len(OKABE)]
        ax.scatter(Z[m, 0], Z[m, 1], s=42, c=c, marker=marcas[t % len(marcas)], alpha=.6,
                   edgecolors="black", linewidths=.4, label=f"Tópico {t + 1} (n={int(m.sum())})")
        ax.text(Z[m, 0].mean(), Z[m, 1].mean(), str(t + 1), fontsize=15, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=.25", fc="white", ec="black", lw=1.4), zorder=5)
    if (asign == -1).any():
        m = asign == -1
        ax.scatter(Z[m, 0], Z[m, 1], s=24, c="#999999", marker="x", label=f"Sin tópico (n={int(m.sum())})")
    ax.set_title(f"Mapa de comentarios por tópico (SVD de TF-IDF) — {nombre}")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="best")
    _reg(f"{pref}_4_mapa", fig)


def g_frecuencias(sub, nombre, color, hatch, pref, A, top=15):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (n, nom) in zip(axs, ((1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas"))):
        t = tabla_terminos(sub["texto_limpio"], n, top)
        if t.empty:
            ax.set_title(f"{nom} (sin datos)")
            ax.axis("off")
            continue
        t = t.sort_values("frecuencia")
        barras = ax.barh(t["termino"], t["frecuencia"], color=color, edgecolor="black", hatch=hatch)
        for b, v in zip(barras, t["frecuencia"]):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {int(v)}", va="center", fontsize=8)
        ax.set_title(nom)
        ax.set_xlabel("Frecuencia")
    fig.suptitle(f"Frecuencia de palabras — {nombre} (n={len(sub)})", fontsize=14, fontweight="bold")
    _reg(f"{pref}_1_frecuencias", fig)

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  (Nube omitida: pip install wordcloud)")
        return
    texto = " ".join(sub["texto_limpio"].dropna().astype(str))
    if not texto.strip():
        return
    nube = WordCloud(width=1200, height=650, background_color="white", max_words=100,
                     collocations=False, random_state=A.semilla,
                     color_func=lambda *a, _c=color, **k: _c).generate(texto)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Nube de palabras — {nombre} (n={len(sub)})", fontsize=15, fontweight="bold")
    _reg(f"{pref}_2_nube", fig)


# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================

def procesar_particion(sub, nombre, color, hatch, pref, A, es_negativa):
    sub = sub.reset_index(drop=True)
    enfoque, motivo = decidir_enfoque(len(sub), A)
    anunciar(nombre, len(sub), enfoque, motivo, A)
    if enfoque == "ninguno":
        sub["topico"] = np.nan
        return sub
    res = modelar(sub, A) if enfoque == "tópicos" else None
    if enfoque == "tópicos" and res is None:
        print("  No se pudo ajustar un modelo de tópicos (vocabulario insuficiente): "
              "se usa frecuencia de palabras.")
        enfoque = "frecuencias"
    if enfoque == "tópicos":
        sub["topico"] = np.where(res["asign"] >= 0, res["asign"] + 1, 0)   # 0 = sin tópico
        nota = es_negativa and not A.dmo_puede_ser_negativa and not sub["es_dmo"].any()
        reportar_topicos(sub, res, nombre, A, nota_gap=nota)
        g_coherencia(res, nombre, pref)
        g_terminos_topicos(res, nombre, color, hatch, pref)
        g_prevalencia_fuentes(sub, res, nombre, A, pref)
        g_mapa(res, nombre, A, pref)
        sub.attrs["terminos"] = res["terminos"]
    else:
        sub["topico"] = np.nan
        reportar_frecuencias(sub, A)
        g_frecuencias(sub, nombre, color, hatch, pref, A)
    return sub



# ============================================================
# PROGRAMA PRINCIPAL (todos los pasos)
# ============================================================


# ============================================================
# 6. COMPARACIÓN DE TÓPICOS: DMO VS IMAGEN PERCIBIDA
# ============================================================

def comparar_topicos_dmo_percibida(normales, A):
    """Ajusta un único NMF sobre todos los comentarios normales para comparar
    la prevalencia de los mismos tópicos entre DMO y fuentes percibidas.
    """
    titulo("COMPARACIÓN DE TÓPICOS: IMAGEN PROYECTADA (DMO) VS PERCIBIDA")

    datos = normales.copy()
    datos["texto_limpio"] = datos["texto_limpio"].fillna("").astype(str)
    datos = datos[datos["texto_limpio"].str.strip().ne("")].reset_index(drop=True)
    dmo = datos[datos["es_dmo"]]
    percibida = datos[~datos["es_dmo"]]

    print(f"\nImagen proyectada (DMO): {len(dmo):,} comentarios")
    print(f"Imagen percibida (todas las demás fuentes): {len(percibida):,} comentarios")
    if dmo.empty or percibida.empty:
        print("No se puede realizar la comparación: uno de los grupos está vacío.")
        return
    if len(datos) < 3:
        print("No hay suficientes comentarios para modelar tópicos.")
        return

    # TF-IDF compartido: mismo vocabulario y misma representación para ambos grupos.
    vectorizador = TfidfVectorizer(
        token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2),
        min_df=2 if len(datos) >= 300 else 1, max_df=0.95,
        sublinear_tf=True
    )
    try:
        X = vectorizador.fit_transform(datos["texto_limpio"])
    except ValueError:
        print("No se pudo construir TF-IDF: vocabulario vacío o insuficiente.")
        return

    if X.shape[1] < 2:
        print("El vocabulario contiene menos de dos términos; no se pueden extraer tópicos.")
        return

    k_solicitado = A.n_topicos if A.n_topicos is not None else 5
    k = min(int(k_solicitado), X.shape[0] - 1, X.shape[1] - 1)
    if k < 2:
        print("No hay suficientes datos o términos para generar al menos dos tópicos.")
        return

    print(f"\nMétodo de comparación: TF-IDF + NMF | Número de tópicos: {k}")
    modelo = NMF(n_components=k, init="nndsvda", random_state=A.semilla,
                 max_iter=600)
    W = modelo.fit_transform(X)
    H = modelo.components_
    asignaciones = W.argmax(axis=1)
    datos["topico_comparacion"] = asignaciones + 1
    datos["grupo_comparacion"] = np.where(datos["es_dmo"], "DMO", "Imagen percibida")

    vocabulario = vectorizador.get_feature_names_out()
    topicos = {}
    print("\nTÉRMINOS REPRESENTATIVOS POR TÓPICO")
    for t in range(k):
        idx = np.argsort(H[t])[::-1][:10]
        topicos[t + 1] = vocabulario[idx].tolist()
        print(f"\nTópico {t + 1}: {', '.join(topicos[t + 1])}")

    # Prevalencia como porcentaje dentro de cada grupo (no porcentajes del total global).
    conteos = pd.crosstab(datos["topico_comparacion"], datos["grupo_comparacion"])
    conteos = conteos.reindex(range(1, k + 1), fill_value=0)
    porcentajes = conteos.astype(float)
    for grupo, total in (("DMO", len(dmo)), ("Imagen percibida", len(percibida))):
        if grupo not in porcentajes.columns:
            porcentajes[grupo] = 0.0
        porcentajes[grupo] = porcentajes[grupo] / total * 100.0

    comparacion = pd.DataFrame({
        "topico": [f"Tópico {i}" for i in range(1, k + 1)],
        "terminos_representativos": [", ".join(topicos[i][:6]) for i in range(1, k + 1)],
        "DMO_pct": porcentajes["DMO"].values,
        "percibida_pct": porcentajes["Imagen percibida"].values,
    })
    comparacion["diferencia_pp_percibida_menos_DMO"] = (
        comparacion["percibida_pct"] - comparacion["DMO_pct"]
    )

    # El 5% es un umbral descriptivo para señalar tópicos con presencia apreciable.
    umbral_presencia = 5.0
    def clasificar_fila(fila):
        d, p = fila["DMO_pct"], fila["percibida_pct"]
        if d >= umbral_presencia and p >= umbral_presencia:
            return "Coincidencia temática"
        if p >= umbral_presencia and d < umbral_presencia:
            return "Brecha: tema de usuarios"
        if d >= umbral_presencia and p < umbral_presencia:
            return "Tema destacado por DMO"
        return "Presencia baja en ambos"

    comparacion["interpretacion"] = comparacion.apply(clasificar_fila, axis=1)
    print("\nCOMPARACIÓN DE TÓPICOS DOMINANTES (porcentaje dentro de cada grupo)")
    print(comparacion.round(2).to_string(index=False))

    for etiqueta, categoria in (
        ("COINCIDENCIAS TEMÁTICAS", "Coincidencia temática"),
        ("BRECHAS: TEMAS DE LOS USUARIOS", "Brecha: tema de usuarios"),
        ("TEMAS DESTACADOS POR LA DMO", "Tema destacado por DMO"),
    ):
        print(f"\n{etiqueta}")
        sub = comparacion[comparacion["interpretacion"] == categoria]
        if sub.empty:
            print("  No se identificaron tópicos en esta categoría.")
        else:
            for _, fila in sub.iterrows():
                print(
                    f"  {fila['topico']}: {fila['terminos_representativos']} | "
                    f"DMO={fila['DMO_pct']:.1f}% | "
                    f"Percibida={fila['percibida_pct']:.1f}% | "
                    f"Diferencia={fila['diferencia_pp_percibida_menos_DMO']:+.1f} pp"
                )

    # Gráfica de barras agrupadas.
    x = np.arange(k)
    ancho = 0.36
    fig, ax = plt.subplots(figsize=(max(11, k * 1.8), 7))
    b1 = ax.bar(x - ancho / 2, comparacion["DMO_pct"], ancho,
                color=C_DMO, edgecolor="black", hatch="///", label="Imagen proyectada (DMO)")
    b2 = ax.bar(x + ancho / 2, comparacion["percibida_pct"], ancho,
                color=C_PERC, edgecolor="black", hatch="...", label="Imagen percibida")
    ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tópico {i}\n{', '.join(topicos[i][:3])}" for i in range(1, k + 1)],
                       rotation=15, ha="right")
    ax.set_ylabel("Porcentaje de comentarios dentro del grupo")
    ax.set_title("Tópicos compartidos: imagen proyectada (DMO) vs imagen percibida")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    _reg("06_comparacion_topicos_DMO_percibida", fig)

    # Distribución por fuente: porcentajes por fila para comparar plataformas.
    por_fuente = pd.crosstab(datos["fuente"].astype(str), datos["topico_comparacion"],
                             normalize="index") * 100.0
    por_fuente = por_fuente.reindex(columns=range(1, k + 1), fill_value=0)
    por_fuente.columns = [f"Tópico {i}" for i in range(1, k + 1)]
    print("\nDISTRIBUCIÓN DE TÓPICOS POR FUENTE (%)")
    print(por_fuente.round(2).to_string())
    for fuente in por_fuente.index:
        if str(fuente) == str(getattr(A, "dmo", "")):
            continue
        fila = por_fuente.loc[fuente]
        dominante = int(np.argmax(fila.values)) + 1
        print(f"\n{fuente}: Tópico {dominante} ({fila.iloc[dominante - 1]:.1f}%)")
        print("  Términos: " + ", ".join(topicos[dominante][:6]))

    fig, ax = plt.subplots(figsize=(max(11, len(por_fuente) * 1.1), 7))
    por_fuente.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Distribución porcentual de tópicos por fuente")
    ax.set_xlabel("Fuente")
    ax.set_ylabel("Porcentaje de comentarios (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Tópicos", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=.2)
    _reg("07_topicos_por_fuente", fig)

    # Se conservan para exportarlos desde el bloque de guardado de main().
    A.comparacion_topicos = comparacion
    A.topicos_por_fuente = por_fuente
    A.comentarios_topicos_comparacion = datos
    print("\nComparación temática finalizada.")


# ============================================================
# 7. COMPARACIONES ADICIONALES: SENTIMIENTO, SIMILITUD Y TÓPICOS BUENOS / MALOS
#    + RESUMEN ESTADÍSTICO (media, mediana, desviación estándar)
# ============================================================
from scipy.stats import mannwhitneyu, ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

RESUMEN = []          # filas del resumen estadístico final


def _desc(x):
    """Estadísticos descriptivos de un vector (ignora NaN). Desviación estándar muestral (ddof=1)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, media=np.nan, mediana=np.nan, desv_est=np.nan,
                    minimo=np.nan, maximo=np.nan)
    return dict(n=int(x.size), media=float(x.mean()), mediana=float(np.median(x)),
                desv_est=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                minimo=float(x.min()), maximo=float(x.max()))


def _resumen(comparacion, metrica, grupo, x):
    RESUMEN.append({"comparacion": comparacion, "metrica": metrica, "grupo": str(grupo),
                    **_desc(x)})


def _orden_fuentes(fuentes, A):
    fuentes = [str(f) for f in fuentes]
    otras = sorted(f for f in set(fuentes) if f != str(A.dmo))
    return ([str(A.dmo)] if str(A.dmo) in fuentes else []) + otras


def _magnitud_cliff(d):
    d = abs(d)
    return ("insignificante" if d < .147 else "pequeña" if d < .33
            else "mediana" if d < .474 else "grande")


# ------------------------------------------------------------
# 7.1 SENTIMIENTO: DMO vs imagen percibida
# ------------------------------------------------------------

def comparar_sentimiento(normales, A):
    """Compara la polaridad continua (-1..+1) del sentimiento híbrido entre la DMO y
    la imagen percibida. Se usa la polaridad numérica real (no la etiqueta forzada
    positiva de la DMO), así la comparación no está sesgada por diseño."""
    titulo("COMPARACIÓN DE SENTIMIENTO: IMAGEN PROYECTADA (DMO) VS PERCIBIDA")
    d = normales[normales["polaridad"].notna()].copy()
    d["fuente"] = d["fuente"].astype(str)
    a = d.loc[d["es_dmo"], "polaridad"].to_numpy(float)
    b = d.loc[~d["es_dmo"], "polaridad"].to_numpy(float)
    print(f"\nComentarios con polaridad: DMO={len(a):,} | percibida={len(b):,}")
    if len(a) < 2 or len(b) < 2:
        print("No se puede comparar: uno de los grupos tiene menos de 2 comentarios con polaridad.")
        return

    u = A.umbral
    filas = []
    for nombre, x in (("DMO", a), ("Imagen percibida", b)):
        filas.append({"nivel": "grupo", "nombre": nombre, **_desc(x),
                      "pct_positivo": 100 * float(np.mean(x >= u))})
    orden = _orden_fuentes(d["fuente"].unique(), A)
    for f in orden:
        x = d.loc[d["fuente"] == f, "polaridad"].to_numpy(float)
        filas.append({"nivel": "fuente", "nombre": f, **_desc(x),
                      "pct_positivo": 100 * float(np.mean(x >= u))})
    tabla = pd.DataFrame(filas)
    print("\nPOLARIDAD (-1 muy negativo ... +1 muy positivo)")
    print(tabla.round(3).to_string(index=False))

    # --- Pruebas y tamaños de efecto -------------------------------------
    U, p_u = mannwhitneyu(b, a, alternative="two-sided")
    delta = 2 * U / (len(a) * len(b)) - 1                 # Cliff: >0 percibida mayor que DMO
    ks, p_ks = ks_2samp(a, b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    cohen = (b.mean() - a.mean()) / sp if sp > 0 else np.nan
    wass = wasserstein_distance(a, b)
    cont = np.array([[(a >= u).sum(), (a < u).sum()], [(b >= u).sum(), (b < u).sum()]])
    try:
        chi, p_chi, _, _ = chi2_contingency(cont)
        v_cramer = float(np.sqrt(chi / cont.sum()))
    except ValueError:
        chi, p_chi, v_cramer = np.nan, np.nan, np.nan

    pruebas = pd.DataFrame([
        ("Diferencia de medias (percibida - DMO)", b.mean() - a.mean(), np.nan),
        ("Diferencia de medianas (percibida - DMO)", np.median(b) - np.median(a), np.nan),
        ("Mann-Whitney U (estadístico)", float(U), p_u),
        ("Delta de Cliff (efecto, -1..1)", delta, np.nan),
        ("d de Cohen (efecto)", cohen, np.nan),
        ("Kolmogorov-Smirnov D", float(ks), p_ks),
        ("Distancia de Wasserstein (unidades de polaridad)", wass, np.nan),
        ("Chi-cuadrada % positivo vs negativo", chi, p_chi),
        ("V de Cramér (efecto)", v_cramer, np.nan),
    ], columns=["prueba", "valor", "p_valor"])
    print("\nPRUEBAS ESTADÍSTICAS DMO vs PERCIBIDA")
    print(pruebas.round(4).to_string(index=False))
    sentido = "más positiva" if b.mean() > a.mean() else "más negativa"
    print(f"\nLectura: la imagen percibida es {sentido} que la proyectada "
          f"(Δ media = {b.mean() - a.mean():+.3f}; efecto {_magnitud_cliff(delta)} según Cliff). "
          f"Con muestras grandes casi todo sale 'significativo': guíate por el tamaño del efecto.")

    # --- Gráfica ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    bins = np.linspace(-1, 1, 31)
    for x, color, hatch, nombre in ((a, C_DMO, H_DMO, "Imagen proyectada (DMO)"),
                                    (b, C_PERC, H_PERC, "Imagen percibida")):
        ax[0].hist(x, bins=bins, density=True, color=color, alpha=.55, edgecolor="black",
                   hatch=hatch, label=f"{nombre} (n={len(x):,})")
        ax[0].axvline(x.mean(), color=color, lw=2.5, ls="-")
        ax[0].axvline(x.mean() - x.std(ddof=1), color=color, lw=1.4, ls="--")
        ax[0].axvline(x.mean() + x.std(ddof=1), color=color, lw=1.4, ls="--")
    ax[0].axvline(u, color="black", lw=1, ls=":")
    ax[0].set_xlim(-1.05, 1.15)
    ax[0].set_xlabel("Polaridad (−1 negativo … +1 positivo)")
    ax[0].set_ylabel("Densidad")
    ax[0].set_title("Distribución de polaridad\n(línea sólida = media; discontinuas = ±1 desv. est.)")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=.25)

    por = tabla[tabla["nivel"] == "fuente"].reset_index(drop=True)
    y = np.arange(len(por))[::-1]
    i_otra = 0
    for yi, (_, r) in zip(y, por.iterrows()):
        es_d = str(r["nombre"]) == str(A.dmo)
        color = C_DMO if es_d else C_FUENTES[i_otra % len(C_FUENTES)]
        i_otra += 0 if es_d else 1
        ax[1].barh(yi, r["media"], xerr=r["desv_est"], color=color, edgecolor="black",
                   hatch=H_DMO if es_d else H_PERC, capsize=4, alpha=.85)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([("DMO: " if str(r["nombre"]) == str(A.dmo) else "") + str(r["nombre"])
                           + f"\n{r['media']:+.2f} ± {r['desv_est']:.2f}"
                           for _, r in por.iterrows()], fontsize=9)
    ax[1].axvline(0, color="black", lw=1)
    ax[1].axvline(u, color="black", lw=1, ls=":")
    ax[1].set_xlim(-1.15, 1.15)
    ax[1].set_xlabel("Polaridad media ± desviación estándar")
    ax[1].set_title("Polaridad por fuente")
    ax[1].grid(axis="x", alpha=.25)
    fig.suptitle("Comparación de sentimiento: DMO vs imagen percibida", fontsize=13)
    _reg("08_comparacion_sentimiento_DMO_percibida", fig)

    # --- Para el resumen / guardado ---------------------------------------
    _resumen("Sentimiento", "polaridad", "DMO", a)
    _resumen("Sentimiento", "polaridad", "Imagen percibida", b)
    for f in orden:
        _resumen("Sentimiento", "polaridad", f, d.loc[d["fuente"] == f, "polaridad"])
    A.sentimiento_por_fuente = tabla
    A.sentimiento_pruebas = pruebas


# ------------------------------------------------------------
# 7.1b PROPORCIONES POSITIVO vs NEGATIVO: DMO vs imagen percibida (global y por tópico)
# ------------------------------------------------------------
# Objetivo: detectar TEMAS que la imagen proyectada (DMO) presenta como positivos pero que
# los usuarios perciben de forma negativa. Se usa la polaridad REAL de cada comentario
# (>= umbral = positivo, < umbral = negativo), no la etiqueta forzada positiva de la DMO.

MIN_N_PROP = 5       # mínimo de comentarios por grupo y tópico para sacar conclusiones
ALFA_PROP = 0.05     # nivel de significancia (con corrección de Holm entre tópicos)


def _estrellas_p(p):
    if not np.isfinite(p):
        return "n/d"
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def _wilson_ic(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _holm(pvals):
    """Corrección de Holm-Bonferroni (controla el error global al probar varios tópicos)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    prev = 0.0
    for rango, i in enumerate(idx[np.argsort(p[idx])]):
        prev = max(prev, min(1.0, (len(idx) - rango) * p[i]))
        out[i] = prev
    return out


def _dos_proporciones(k1, n1, k2, n2):
    """Prueba bilateral H0: p1 = p2 (grupo 1 = DMO, grupo 2 = percibida).
    Devuelve (diferencia p2-p1, IC95 Newcombe, p-valor, método, h de Cohen).
    z agrupada; si alguna frecuencia esperada < 5 usa la prueba exacta de Fisher."""
    from scipy.stats import fisher_exact, norm
    p1, p2 = k1 / n1, k2 / n2
    dif = p2 - p1
    l1, u1 = _wilson_ic(k1, n1)
    l2, u2 = _wilson_ic(k2, n2)
    ic = (dif - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2),
          dif + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    pp = (k1 + k2) / (n1 + n2)
    if pp in (0.0, 1.0):
        return dif, ic, 1.0, "sin variación", h
    esperadas = (n1 * pp, n1 * (1 - pp), n2 * pp, n2 * (1 - pp))
    if min(esperadas) < 5:
        _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
        return dif, ic, float(p), "Fisher exacta", h
    z = dif / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return dif, ic, float(2 * norm.sf(abs(z))), "z de 2 proporciones", h


def _binomial_pos_neg(k_neg, n):
    """Prueba binomial exacta bilateral dentro de UN grupo: H0 = 50 % positivo / 50 % negativo."""
    from scipy.stats import binom
    if n <= 0:
        return np.nan
    pmf = binom.pmf(np.arange(n + 1), n, 0.5)
    return float(min(1.0, pmf[pmf <= pmf[k_neg] * (1 + 1e-7)].sum()))


def _magnitud_h(h):
    h = abs(h)
    return ("insignificante" if h < .2 else "pequeño" if h < .5
            else "mediano" if h < .8 else "grande")


def _moda_polaridad(x, ancho=0.1):
    """Moda de una variable continua: centro del intervalo (de 'ancho') con más comentarios.
    En empates se devuelve el intervalo más negativo."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    bordes = np.arange(-1.0, 1.0 + ancho / 2, ancho)
    cuentas, bordes = np.histogram(x, bins=bordes)
    i = int(cuentas.argmax())
    return float((bordes[i] + bordes[i + 1]) / 2)


def _desc_prop(x, u):
    """Frecuencia, proporción, media, mediana y moda de la polaridad de un grupo."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    k_neg = int((x < u).sum())
    k_pos = n - k_neg
    return {"n": n, "frec_positivo": k_pos, "frec_negativo": k_neg,
            "pct_positivo": 100 * k_pos / n if n else np.nan,
            "pct_negativo": 100 * k_neg / n if n else np.nan,
            "media": float(x.mean()) if n else np.nan,
            "mediana": float(np.median(x)) if n else np.nan,
            "moda": _moda_polaridad(x),
            "moda_clase": ("positivo" if k_pos > k_neg else "negativo" if k_neg > k_pos
                           else "empate") if n else "n/d",
            "p_binomial_pos_vs_neg": _binomial_pos_neg(k_neg, n)}


def _clasificar_brecha(r):
    if r["n_DMO"] < MIN_N_PROP or r["n_percibida"] < MIN_N_PROP:
        return "Muestra insuficiente"
    dmo_pos = r["pct_positivo_DMO"] > 50
    perc_neg = r["pct_negativo_percibida"] > 50
    sig = bool(r["p_holm"] < ALFA_PROP)
    if dmo_pos and perc_neg:
        return ("BRECHA: proyectada positiva, percibida negativa" if sig
                else "Posible brecha (no significativa)")
    if sig and r["dif_neg_pp"] > 0:
        return "Percibida más negativa que la DMO"
    if sig and r["dif_neg_pp"] < 0:
        return "Percibida más positiva que la DMO"
    return "Positiva en ambas" if dmo_pos else "Negativa en ambas"


def comparar_proporciones_sentimiento(normales, A):
    """Pruebas de proporciones positivo vs negativo entre la DMO y la imagen percibida.
    1) Global y por fuente. 2) Por tópico (los tópicos NMF compartidos de la sección 6).
    Incluye frecuencia, media, mediana y moda, y una gráfica con los resultados."""
    titulo("PRUEBAS DE PROPORCIONES: POSITIVO VS NEGATIVO, IMAGEN PROYECTADA (DMO) VS PERCIBIDA")
    u = A.umbral
    d = normales[normales["polaridad"].notna()].copy()
    d["fuente"] = d["fuente"].astype(str)
    d["es_dmo"] = d["es_dmo"].astype(bool)
    a, b = d.loc[d["es_dmo"], "polaridad"].to_numpy(float), \
        d.loc[~d["es_dmo"], "polaridad"].to_numpy(float)
    print(f"\nSe usa la polaridad real de cada comentario (>= {u:g} positivo, < {u:g} negativo); "
          f"se excluyen los comentarios 'sin señal'.")
    print(f"Comentarios con polaridad: DMO={len(a):,} | percibida={len(b):,}")
    if len(a) < 2 or len(b) < 2:
        print("No se puede comparar: uno de los grupos tiene menos de 2 comentarios con polaridad.")
        return

    # ---------------- 1) GLOBAL Y POR FUENTE ----------------
    grupos = [("DMO", a), ("Imagen percibida", b)] + [
        (f, d.loc[d["fuente"] == f, "polaridad"].to_numpy(float))
        for f in _orden_fuentes(d["fuente"].unique(), A)]
    glob = pd.DataFrame([{"grupo": n, **_desc_prop(x, u)} for n, x in grupos])
    print("\nFRECUENCIA, MEDIA, MEDIANA Y MODA DE LA POLARIDAD")
    print(glob.round(3).to_string(index=False))

    pr = [("Imagen percibida (todas)", b)] + [
        (n, x) for n, x in grupos[2:] if n != str(A.dmo)]
    filas = []
    for n, x in pr:
        k1, k2 = int((a < u).sum()), int((x < u).sum())
        dif, ic, p, met, h = _dos_proporciones(k1, len(a), k2, len(x))
        filas.append({"comparacion": f"{n} vs DMO", "pct_neg_DMO": 100 * k1 / len(a),
                      "pct_neg_grupo": 100 * k2 / len(x), "dif_neg_pp": 100 * dif,
                      "IC95_inf_pp": 100 * ic[0], "IC95_sup_pp": 100 * ic[1], "p_valor": p,
                      "metodo": met, "h_cohen": h})
    pruebas_g = pd.DataFrame(filas)
    pruebas_g["p_holm"] = _holm(pruebas_g["p_valor"])
    pruebas_g["efecto"] = pruebas_g["h_cohen"].map(_magnitud_h)
    print("\nPRUEBA DE 2 PROPORCIONES: % NEGATIVO de cada grupo vs % NEGATIVO de la DMO "
          "(dif > 0: más negativo que la DMO)")
    print(pruebas_g.round(4).to_string(index=False))
    A.prop_sent_global = glob
    A.prop_sent_pruebas_global = pruebas_g

    # ---------------- 2) POR TÓPICO ----------------
    base = getattr(A, "comentarios_topicos_comparacion", None)
    if base is None or "topico_comparacion" not in base.columns:
        print("\nNo hay tópicos compartidos calculados: solo se muestra la comparación global.")
        return
    base = base[base["polaridad"].notna()].copy()
    base["es_dmo"] = base["es_dmo"].astype(bool)
    comp = getattr(A, "comparacion_topicos", None)
    terminos = (dict(zip(comp["topico"], comp["terminos_representativos"]))
                if comp is not None else {})
    topicos = sorted(base["topico_comparacion"].unique())

    desc_filas, test_filas = [], []
    for t in topicos:
        sub = base[base["topico_comparacion"] == t]
        xd = sub.loc[sub["es_dmo"], "polaridad"].to_numpy(float)
        xp = sub.loc[~sub["es_dmo"], "polaridad"].to_numpy(float)
        nombre = f"Tópico {t}"
        term = terminos.get(nombre, "")
        for gname, x in (("DMO", xd), ("Imagen percibida", xp)):
            desc_filas.append({"topico": nombre, "terminos": term, "grupo": gname,
                               **_desc_prop(x, u)})
            if len(x) >= 2:
                _resumen("Sentimiento por tópico", f"polaridad · {nombre}", gname, x)
        fila = {"topico": nombre, "terminos": term, "n_DMO": len(xd), "n_percibida": len(xp)}
        dd, dp = _desc_prop(xd, u), _desc_prop(xp, u)
        fila.update({"pct_positivo_DMO": dd["pct_positivo"], "pct_negativo_DMO": dd["pct_negativo"],
                     "pct_positivo_percibida": dp["pct_positivo"],
                     "pct_negativo_percibida": dp["pct_negativo"]})
        if len(xd) and len(xp):
            dif, ic, p, met, h = _dos_proporciones(dd["frec_negativo"], len(xd),
                                                   dp["frec_negativo"], len(xp))
            fila.update({"dif_neg_pp": 100 * dif, "IC95_inf_pp": 100 * ic[0],
                         "IC95_sup_pp": 100 * ic[1], "p_valor": p, "metodo": met,
                         "h_cohen": h})
        else:
            fila.update({"dif_neg_pp": np.nan, "IC95_inf_pp": np.nan, "IC95_sup_pp": np.nan,
                         "p_valor": np.nan, "metodo": "sin datos en un grupo", "h_cohen": np.nan})
        test_filas.append(fila)
    desc_t = pd.DataFrame(desc_filas)
    tabla = pd.DataFrame(test_filas)
    tabla["p_holm"] = _holm(tabla["p_valor"])
    tabla["efecto"] = tabla["h_cohen"].map(lambda h: _magnitud_h(h) if np.isfinite(h) else "n/d")
    tabla["interpretacion"] = tabla.apply(_clasificar_brecha, axis=1)

    print("\nPOR TÓPICO: FRECUENCIA, MEDIA, MEDIANA Y MODA DE LA POLARIDAD")
    print(desc_t.drop(columns="terminos").round(3).to_string(index=False))
    print("\nPOR TÓPICO: PRUEBA DE 2 PROPORCIONES SOBRE EL % NEGATIVO "
          f"(Holm entre {len(tabla)} tópicos; efecto = h de Cohen)")
    cols = ["topico", "n_DMO", "n_percibida", "pct_negativo_DMO", "pct_negativo_percibida",
            "dif_neg_pp", "IC95_inf_pp", "IC95_sup_pp", "p_valor", "p_holm", "metodo",
            "efecto", "interpretacion"]
    print(tabla[cols].round(4).to_string(index=False))
    A.prop_sent_descriptivos_topicos = desc_t
    A.prop_sent_topicos = tabla

    brechas = tabla[tabla["interpretacion"].str.startswith("BRECHA")]
    posibles = tabla[tabla["interpretacion"].str.startswith("Posible")]
    print("\nLECTURA: TEMAS POSITIVOS EN LA IMAGEN PROYECTADA PERO NEGATIVOS EN LA PERCIBIDA")
    if brechas.empty:
        print("  Ningún tópico cumple a la vez: mayoría positiva en la DMO, mayoría negativa en la "
              "percibida y diferencia significativa.")
    for _, r in brechas.iterrows():
        print(f"  {r['topico']} ({r['terminos']}): DMO {r['pct_positivo_DMO']:.0f}% positivo -> "
              f"percibida {r['pct_negativo_percibida']:.0f}% negativo "
              f"(Δ negativo = {r['dif_neg_pp']:+.1f} pp, IC95 [{r['IC95_inf_pp']:+.1f}, "
              f"{r['IC95_sup_pp']:+.1f}], p Holm = {r['p_holm']:.4f}, efecto {r['efecto']}).")
    for _, r in posibles.iterrows():
        print(f"  (posible, sin significancia) {r['topico']} ({r['terminos']}): "
              f"n DMO={r['n_DMO']}, n percibida={r['n_percibida']}.")
    print("\nNotas: (1) el sentimiento es del comentario completo y el tópico es el dominante del "
          "NMF, así que es una aproximación a 'sentimiento por tema', no un análisis por aspecto. "
          f"(2) Con menos de {MIN_N_PROP} comentarios en un grupo no se concluye. (3) Con muestras "
          "grandes todo sale significativo: revisa el IC y la h de Cohen. (4) La moda de la "
          "polaridad es el centro del intervalo de 0.1 más poblado.")

    _grafica_proporciones(tabla, pruebas_g, A)


def _grafica_proporciones(tabla, pruebas_g, A):
    """Izquierda: % negativo (hacia la izquierda) vs % positivo (hacia la derecha) por tópico y
    grupo. Derecha: diferencia en % negativo (percibida − DMO) con IC95 y significancia."""
    k = len(tabla)
    fig, ax = plt.subplots(1, 2, figsize=(17, max(6.5, 1.15 * k + 3.2)),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    etiquetas, ypos, y = [], [], 0.0
    filas = [("GLOBAL", None)] + [(r["topico"], r) for _, r in tabla.iterrows()]
    g_row = pruebas_g.iloc[0]
    for nombre, r in filas[::-1]:
        if r is None:
            dmo_neg, per_neg = g_row["pct_neg_DMO"], g_row["pct_neg_grupo"]
            dmo_pos, per_pos = 100 - dmo_neg, 100 - per_neg
            etq = "GLOBAL"
        else:
            dmo_neg, per_neg = r["pct_negativo_DMO"], r["pct_negativo_percibida"]
            dmo_pos, per_pos = r["pct_positivo_DMO"], r["pct_positivo_percibida"]
            term = ", ".join(str(r["terminos"]).split(", ")[:2])
            etq = f"{nombre}\n{textwrap.shorten(term, 24, placeholder='…')}"
        for off, neg, pos, hatch_c, lab in ((.19, dmo_neg, dmo_pos, H_DMO, "DMO"),
                                             (-.19, per_neg, per_pos, H_PERC, "Percibida")):
            if not (np.isfinite(neg) and np.isfinite(pos)):
                continue
            ax[0].barh(y + off, -neg, .34, color=C_NEG, edgecolor="black", hatch="xx", alpha=.9)
            ax[0].barh(y + off, pos, .34, color=C_POS, edgecolor="black", hatch="\\\\", alpha=.9)
            ax[0].text(-neg - 1.2, y + off, f"{neg:.0f}%", ha="right", va="center", fontsize=8)
            ax[0].text(pos + 1.2, y + off, f"{pos:.0f}%", ha="left", va="center", fontsize=8)
            ax[0].text(-134, y + off, lab, ha="left", va="center", fontsize=8, style="italic")
        etiquetas.append(etq)
        ypos.append(y)
        y += 1.0
    ax[0].axvline(0, color="black", lw=1)
    ax[0].set_yticks(ypos)
    ax[0].set_yticklabels(etiquetas, fontsize=8.5)
    ax[0].set_xlim(-136, 118)
    ax[0].set_xticks(np.arange(-100, 101, 25))
    ax[0].set_xticklabels([f"{abs(v)}%" for v in np.arange(-100, 101, 25)])
    ax[0].set_xlabel("◄ % negativo        |        % positivo ►")
    ax[0].set_title("Negativo vs positivo por tópico\n(barra de arriba = DMO, de abajo = percibida)")
    ax[0].legend(handles=[Patch(fc=C_NEG, ec="black", hatch="xx", label="Negativo"),
                          Patch(fc=C_POS, ec="black", hatch="\\\\", label="Positivo")],
                 loc="lower right", fontsize=9)
    ax[0].grid(axis="x", alpha=.25)

    # --- forest plot de la diferencia en % negativo ---
    y = 0.0
    for nombre, r in filas[::-1]:
        if r is None:
            dif, lo, hi = g_row["dif_neg_pp"], g_row["IC95_inf_pp"], g_row["IC95_sup_pp"]
            p, inter = g_row["p_valor"], "global"
        else:
            dif, lo, hi, p, inter = (r["dif_neg_pp"], r["IC95_inf_pp"], r["IC95_sup_pp"],
                                     r["p_holm"], r["interpretacion"])
        if np.isfinite(dif):
            brecha = str(inter).startswith("BRECHA")
            color = C_NEG if brecha else ("#999999" if not (np.isfinite(p) and p < ALFA_PROP)
                                          else C_PERC)
            ax[1].errorbar(dif, y, xerr=[[dif - lo], [hi - dif]], fmt="X" if brecha else "o",
                           color=color, ecolor="black", capsize=4, ms=10 if brecha else 8,
                           mec="black", zorder=3)
            ax[1].text(max(hi, dif) + 2, y, _estrellas_p(p), va="center", fontsize=9,
                       fontweight="bold" if brecha else "normal")
        y += 1.0
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels([""] * len(ypos))
    ax[1].set_ylim(ax[0].get_ylim())
    lim = max(10, np.nanmax(np.abs(pd.concat([tabla["IC95_inf_pp"], tabla["IC95_sup_pp"]]))) + 12)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_xlabel("Δ % negativo (percibida − DMO), puntos porcentuales ± IC95")
    ax[1].set_title("¿La percibida es más negativa que la proyectada?\n"
                    "(gris = n.s.; naranja = significativo; X roja = brecha;\n"
                    "* p<.05, ** p<.01, *** p<.001; p de Holm por tópico)")
    ax[1].grid(axis="x", alpha=.25)
    fig.suptitle("Proporciones positivo/negativo: imagen proyectada (DMO) vs imagen percibida",
                 fontsize=13)
    _reg("13_proporciones_sentimiento_topicos", fig)


# ------------------------------------------------------------
# 7.2 SIMILITUD: DMO vs imagen percibida
# ------------------------------------------------------------

def _centroide(X):
    c = np.asarray(X.mean(axis=0)).ravel()
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def _sim_interna_dmo(Xd):
    """Similitud coseno de cada comentario DMO con el centroide de los DEMÁS comentarios DMO
    (leave-one-out), para no inflar la referencia comparando un comentario consigo mismo."""
    S = np.asarray(Xd.sum(axis=0)).ravel()
    xs = np.asarray(Xd @ S).ravel()
    xx = np.asarray(Xd.multiply(Xd).sum(axis=1)).ravel() if hasattr(Xd, "multiply") \
        else (Xd * Xd).sum(axis=1)
    norma2 = np.maximum(float(S @ S) - 2 * xs + xx, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (xs - xx) / np.sqrt(norma2)
    return np.where(np.isfinite(out), out, 0.0)


def _similitud_espacio(Xn, es_dmo, fuentes, A, n_boot):
    """Medidas de similitud en un espacio vectorial con filas de norma 1."""
    rng = np.random.default_rng(A.semilla)
    idx_d, idx_p = np.where(es_dmo)[0], np.where(~es_dmo)[0]
    cd, cp = _centroide(Xn[idx_d]), _centroide(Xn[idx_p])
    res = {"global": float(cd @ cp)}

    # Bootstrap de muestras del MISMO tamaño m: DMO-vs-percibida y DMO-vs-DMO (línea base).
    m = max(1, min(len(idx_d) // 2, len(idx_p)))
    sim_dp, sim_dd, indice = [], [], []
    for _ in range(n_boot):
        perm = rng.permutation(idx_d)
        a_, b_ = perm[:m], perm[m:2 * m]
        c_ = rng.choice(idx_p, m, replace=False)
        ca = _centroide(Xn[a_])
        s_dp = float(ca @ _centroide(Xn[c_]))
        s_dd = float(ca @ _centroide(Xn[b_])) if len(b_) else np.nan
        sim_dp.append(s_dp)
        sim_dd.append(s_dd)
        indice.append(s_dp / s_dd if s_dd and s_dd > 0 else np.nan)
    res["boot_dmo_perc"], res["boot_dmo_dmo"], res["boot_indice"] = (
        np.array(sim_dp), np.array(sim_dd), np.array(indice))
    res["m_boot"] = m

    return res, cd


def comparar_similitud(normales, A):
    """Similitud semántica DMO vs percibida: global (índice de alineación), por fuente
    (matriz fuente x fuente) y por comentario. Espacios: TF-IDF, LSA y (opcional) embeddings."""
    titulo("SIMILITUD: IMAGEN PROYECTADA (DMO) VS IMAGEN PERCIBIDA")
    datos = normales.copy()
    datos["texto_limpio"] = datos["texto_limpio"].fillna("").astype(str)
    datos = datos[datos["texto_limpio"].str.strip().ne("")].reset_index(drop=True)
    datos["fuente"] = datos["fuente"].astype(str)
    es_dmo = datos["es_dmo"].to_numpy(bool)
    if es_dmo.sum() < 4 or (~es_dmo).sum() < 2:
        print("No se puede calcular la similitud: faltan comentarios en la DMO (mín. 4) o en la percibida.")
        return
    print(f"\nDMO: {int(es_dmo.sum()):,} comentarios | percibida: {int((~es_dmo).sum()):,}")

    # ---- Espacios vectoriales -------------------------------------------
    espacios = {}
    try:
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2),
                              min_df=2 if len(datos) >= 300 else 1, max_df=0.95, sublinear_tf=True)
        X = vec.fit_transform(datos["texto_limpio"]).tocsr()
        espacios["TF-IDF (palabras)"] = X
        k_svd = min(100, X.shape[1] - 1, X.shape[0] - 1)
        if k_svd >= 2:
            Z = TruncatedSVD(n_components=k_svd, random_state=A.semilla).fit_transform(X)
            espacios["LSA (semántica latente)"] = normalize(Z)
    except ValueError:
        print("No se pudo construir TF-IDF (vocabulario vacío).")
        return
    if getattr(A, "similitud_embeddings", False):
        try:
            from sentence_transformers import SentenceTransformer
            textos = [_texto_para_bert(t) for t in datos["texto_original"]]
            E = SentenceTransformer(A.bt_modelo).encode(textos, show_progress_bar=False)
            espacios["Embeddings multilingües"] = normalize(np.asarray(E))
        except Exception as e:
            print(f"AVISO: no se pudieron calcular embeddings ({type(e).__name__}). Se omiten.")

    orden = _orden_fuentes(datos["fuente"].unique(), A)
    n_boot = int(getattr(A, "n_boot", 200))
    filas_globales, filas_fuente, matrices, por_comentario = [], [], {}, {}

    for nombre_esp, Xn in espacios.items():
        res, cd = _similitud_espacio(Xn, es_dmo, orden, A, n_boot)
        b_dp, b_dd, b_ix = res["boot_dmo_perc"], res["boot_dmo_dmo"], res["boot_indice"]
        ix = float(np.nanmean(b_ix))
        nivel = "alta" if ix >= .8 else "media" if ix >= .5 else "baja"
        filas_globales.append({
            "espacio": nombre_esp, "similitud_global_DMO_percibida": res["global"],
            "boot_DMO_percibida_media": float(np.nanmean(b_dp)),
            "boot_DMO_percibida_desv": float(np.nanstd(b_dp, ddof=1)),
            "boot_linea_base_DMO_DMO_media": float(np.nanmean(b_dd)),
            "boot_linea_base_DMO_DMO_desv": float(np.nanstd(b_dd, ddof=1)),
            "indice_alineacion_media": ix,
            "indice_alineacion_desv": float(np.nanstd(b_ix, ddof=1)),
            "alineacion": nivel, "tam_muestra_bootstrap": res["m_boot"]})
        _resumen(f"Similitud [{nombre_esp}]", "similitud DMO-percibida (bootstrap)", "global", b_dp)
        _resumen(f"Similitud [{nombre_esp}]", "línea base DMO-DMO (bootstrap)", "global", b_dd)
        _resumen(f"Similitud [{nombre_esp}]", "índice de alineación (bootstrap)", "global", b_ix)

        # Matriz de centroides fuente x fuente y similitud por comentario con el centroide DMO.
        cents = {f: _centroide(Xn[np.where(datos["fuente"].to_numpy() == f)[0]]) for f in orden}
        M = np.array([[float(cents[f1] @ cents[f2]) for f2 in orden] for f1 in orden])
        matrices[nombre_esp] = pd.DataFrame(M, index=orden, columns=orden)
        comentarios = {}
        for f in orden:
            idx = np.where(datos["fuente"].to_numpy() == f)[0]
            if f == str(A.dmo):
                s = _sim_interna_dmo(Xn[idx])
            else:
                s = np.asarray(Xn[idx] @ cd).ravel()
            comentarios[f] = s
            _resumen(f"Similitud [{nombre_esp}]", "coseno comentario→centroide DMO", f, s)
            filas_fuente.append({"espacio": nombre_esp, "fuente": f,
                                 "sim_centroide_vs_DMO": float(cents[f] @ cents[str(A.dmo)]),
                                 **{k: v for k, v in _desc(s).items()}})
        por_comentario[nombre_esp] = comentarios

    glob = pd.DataFrame(filas_globales)
    por_f = pd.DataFrame(filas_fuente)
    print("\nSIMILITUD GLOBAL DMO vs PERCIBIDA (coseno entre centroides; 1 = idénticas)")
    print(glob.round(4).to_string(index=False))
    print("\nÍndice de alineación = similitud(DMO, percibida) / similitud(DMO, DMO) con muestras del "
          "mismo tamaño.\n  1.0 = la percibida se parece a la DMO tanto como la DMO a sí misma "
          "(≥0.8 alta, 0.5–0.8 media, <0.5 baja; cortes orientativos).")
    print("\nSIMILITUD POR FUENTE (comentario→centroide DMO; la DMO es leave-one-out)")
    print(por_f.round(4).to_string(index=False))
    for nombre_esp, M in matrices.items():
        print(f"\nMATRIZ DE SIMILITUD ENTRE FUENTES — {nombre_esp}")
        print(M.round(3).to_string())

    # ---- Gráficas -------------------------------------------------------
    n_e = len(espacios)
    fig, axs = plt.subplots(1, n_e, figsize=(6.4 * n_e, 5.6), squeeze=False)
    for ax, (nombre_esp, M) in zip(axs[0], matrices.items()):
        vmin_m = min(0, M.values.min())
        im = ax.imshow(M.values, cmap=CMAP, vmin=vmin_m, vmax=1)
        ax.set_xticks(range(len(orden)))
        ax.set_yticks(range(len(orden)))
        ax.set_xticklabels(orden, rotation=35, ha="right")
        ax.set_yticklabels(orden)
        for i in range(len(orden)):
            for j in range(len(orden)):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color=_color_texto((v - vmin_m) / (1 - vmin_m)))
        ax.set_title(nombre_esp)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Similitud coseno")
    fig.suptitle("Similitud entre fuentes (centroides)", fontsize=13)
    _reg("09_similitud_matriz_fuentes", fig)

    fig, axs = plt.subplots(1, n_e, figsize=(6.4 * n_e, 5.8), squeeze=False)
    for ax, (nombre_esp, comentarios) in zip(axs[0], por_comentario.items()):
        datos_box = [comentarios[f] for f in orden]
        bp = ax.boxplot(datos_box, positions=range(len(orden)), patch_artist=True,
                        showmeans=True, widths=.6,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
        i_otra = 0
        for caja, f in zip(bp["boxes"], orden):
            es_d = f == str(A.dmo)
            caja.set(facecolor=C_DMO if es_d else C_FUENTES[i_otra % len(C_FUENTES)],
                     hatch=H_DMO if es_d else H_PERC, alpha=.85)
            i_otra += 0 if es_d else 1
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels([("DMO\n(interna)" if f == str(A.dmo) else f) for f in orden],
                           rotation=25, ha="right")
        ax.set_ylabel("Coseno del comentario con el centroide DMO")
        ax.set_title(f"{nombre_esp}\n(rombo = media)")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Similitud por comentario con el centroide de la DMO", fontsize=13)
    _reg("10_similitud_por_comentario", fig)

    fig, ax = plt.subplots(figsize=(max(7, 3.2 * n_e), 5.6))
    x = np.arange(n_e)
    w = .36
    b1 = ax.bar(x - w / 2, glob["boot_DMO_percibida_media"], w,
                yerr=glob["boot_DMO_percibida_desv"], capsize=4, color=C_PERC,
                edgecolor="black", hatch=H_PERC, label="DMO vs percibida")
    b2 = ax.bar(x + w / 2, glob["boot_linea_base_DMO_DMO_media"], w,
                yerr=glob["boot_linea_base_DMO_DMO_desv"], capsize=4, color=C_DMO,
                edgecolor="black", hatch=H_DMO, label="DMO vs DMO (línea base)")
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
    for xi, (_, r) in zip(x, glob.iterrows()):
        ax.text(xi, -0.07, f"índice = {r['indice_alineacion_media']:.2f}\n({r['alineacion']})",
                ha="center", va="top", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(glob["espacio"])
    ax.set_ylabel("Similitud coseno entre centroides (media ± desv. est. bootstrap)")
    ax.set_title("Índice de alineación DMO – imagen percibida")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.subplots_adjust(bottom=.2)
    _reg("11_indice_alineacion", fig)

    A.similitud_global = glob
    A.similitud_por_fuente = por_f
    A.similitud_matrices = matrices


# ------------------------------------------------------------
# 7.3 TÓPICOS BUENOS (partición positiva) vs MALOS (partición negativa)
# ------------------------------------------------------------

def _prevalencia_particion(g):
    """Tabla de prevalencia de tópicos de una partición, por grupo (DMO / percibida)."""
    if g is None or "topico" not in g.columns or g["topico"].isna().all():
        return None
    terminos = g.attrs.get("terminos", [])
    h = g[g["topico"] >= 1]
    if h.empty:
        return None
    k = max(len(terminos), int(h["topico"].max()))
    n_d, n_p = int(h["es_dmo"].sum()), int((~h["es_dmo"]).sum())
    filas = []
    for t in range(1, k + 1):
        ht = h[h["topico"] == t]
        c_d, c_p = int(ht["es_dmo"].sum()), int((~ht["es_dmo"]).sum())
        filas.append({
            "topico": f"Tópico {t}",
            "terminos_representativos": ", ".join(w for w, _ in terminos[t - 1][:6])
            if t - 1 < len(terminos) else "",
            "n_DMO": c_d, "n_percibida": c_p,
            "DMO_pct": 100 * c_d / n_d if n_d else np.nan,
            "percibida_pct": 100 * c_p / n_p if n_p else np.nan})
    tabla = pd.DataFrame(filas)
    tabla["diferencia_pp_percibida_menos_DMO"] = tabla["percibida_pct"] - tabla["DMO_pct"]
    return tabla


def comparar_topicos_buenos_malos(pos, neg, A):
    """Compara los tópicos de la partición positiva ('buenos') y de la negativa ('malos'):
    cuánto pesa cada tópico en la DMO y en la imagen percibida dentro de cada partición.
    Nota: los tópicos de cada partición vienen de modelos distintos (Tópico 1 de positivos
    NO es el Tópico 1 de negativos)."""
    titulo("COMPARACIÓN DE TÓPICOS BUENOS (POSITIVOS) Y MALOS (NEGATIVOS)")
    paneles = []
    for nombre, g, color, hatch, clave in (
            ("Tópicos buenos (comentarios positivos)", pos, C_POS, H_POS, "positivos"),
            ("Tópicos malos (comentarios negativos)", neg, C_NEG, H_NEG, "negativos")):
        tabla = _prevalencia_particion(g)
        print(f"\n{nombre.upper()}")
        if tabla is None:
            print("  Sin modelo de tópicos en esta partición (partición vacía o bajo el umbral: "
                  "se analizó por frecuencia de palabras).")
            paneles.append((nombre, None, color, hatch, clave))
            continue
        con_dmo = tabla["n_DMO"].sum() > 0
        print(tabla.round(2).to_string(index=False))
        if con_dmo:
            p_d = tabla["n_DMO"].to_numpy(float) / tabla["n_DMO"].sum()
            p_p = tabla["n_percibida"].to_numpy(float) / max(tabla["n_percibida"].sum(), 1)
            js = float(jensenshannon(p_d, p_p, base=2)) if p_p.sum() > 0 else np.nan
            print(f"\n  Distancia Jensen-Shannon DMO vs percibida = {js:.3f} "
                  "(0 = mismos pesos temáticos, 1 = totalmente distintos)")
            tabla.attrs["js"] = js
        else:
            print("\n  La partición no contiene comentarios de la DMO: todos los tópicos son "
                  "BRECHAS (lo dicen los usuarios y la DMO no lo comunica).")
        columnas = [("percibida_pct", "% percibida por tópico")]
        if con_dmo:
            columnas = [("DMO_pct", "% DMO por tópico")] + columnas + [
                ("diferencia_pp_percibida_menos_DMO", "diferencia (pp) por tópico")]
        for col, etiqueta in columnas:
            _resumen(f"Tópicos {clave}", etiqueta, "entre tópicos", tabla[col])
        if con_dmo:
            _resumen(f"Tópicos {clave}", "|diferencia| (pp) por tópico", "entre tópicos",
                     tabla["diferencia_pp_percibida_menos_DMO"].abs())
        if "js" in tabla.attrs:
            _resumen(f"Tópicos {clave}", "distancia Jensen-Shannon", "DMO vs percibida", [tabla.attrs["js"]])
        paneles.append((nombre, tabla, color, hatch, clave))
        setattr(A, f"comparacion_topicos_{clave}", tabla)

    if all(p[1] is None for p in paneles):
        return
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.6))
    for ax, (nombre, tabla, color, hatch, clave) in zip(axs, paneles):
        if tabla is None:
            ax.text(.5, .5, "Sin modelo de tópicos\nen esta partición", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(nombre)
            ax.axis("off")
            continue
        k = len(tabla)
        x = np.arange(k)
        con_dmo = tabla["n_DMO"].sum() > 0
        if con_dmo:
            w = .38
            b1 = ax.bar(x - w / 2, tabla["DMO_pct"], w, color=C_DMO, edgecolor="black",
                        hatch=H_DMO, label="Imagen proyectada (DMO)")
            b2 = ax.bar(x + w / 2, tabla["percibida_pct"], w, color=C_PERC, edgecolor="black",
                        hatch=H_PERC, label="Imagen percibida")
            ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8)
            ax.legend()
        else:
            b2 = ax.bar(x, tabla["percibida_pct"], .6, color=color, edgecolor="black",
                        hatch=hatch, label="Imagen percibida (sin DMO en esta partición)")
            ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8)
            ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"T{i + 1}\n" + "\n".join(textwrap.wrap(
                ", ".join(str(tabla.iloc[i]["terminos_representativos"]).split(", ")[:2]), 14))
             for i in range(k)], fontsize=8)
        ax.set_ylabel("% de comentarios del grupo dentro de la partición")
        ax.set_title(nombre)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Tópicos buenos vs malos: peso en la DMO y en la imagen percibida", fontsize=13)
    _reg("12_comparacion_topicos_buenos_malos", fig)


# ------------------------------------------------------------
# 7.5 ANÁLISIS ESPECÍFICO DEL CONCEPTO "PRECIO / VALOR / COSTO"
# ------------------------------------------------------------
# 1) Concepto sintético (vector medio de frases ancla multilingües) y similitud coseno de
#    CADA comentario con él  -> comentarios relacionados.
# 2) Scatter plot interactivo (matplotlib): mapa semántico y similitud vs. polaridad.
# 3) DMO vs. fuentes percibidas: frecuencia, tono y contexto.
# 4) Los N comentarios más cercanos al concepto, con su fuente.

G_DMO, G_PERC = "DMO (proyectada)", "Imagen percibida"

# El concepto se define con frases en varios idiomas; el promedio de sus embeddings es el
# "vector del concepto". Edita la lista para afinar el concepto.
ANCLAS_PRECIO = [
    "precio", "costo", "valor", "cuánto cuesta", "es caro", "es barato", "económico",
    "vale la pena", "buena relación calidad-precio", "tarifa, boleto y entrada",
    "gastar dinero y presupuesto", "demasiado caro, abusivo, cobran de más",
    "price", "cost", "value for money", "expensive", "cheap and affordable",
    "worth the money", "overpriced", "preço", "custo-benefício", "caro e barato",
    "prix et coût", "preis und kosten", "prezzo e costo",
]

# Facetas (contexto) con las que se habla del concepto.
FACETAS_PRECIO = {
    "Costo monetario": ["cuánto cuesta, precio del boleto, tarifa de entrada, pesos",
                        "ticket price, entrance fee, how much it costs"],
    "Valor / calidad-precio": ["vale la pena el precio, buena relación calidad-precio",
                               "worth the money, good value for money"],
    "Accesible / barato": ["barato, económico, accesible, precios justos",
                           "cheap, affordable, budget friendly"],
    "Caro / excesivo": ["muy caro, costoso, precios excesivos, abusivos, trampa para turistas",
                        "expensive, overpriced, rip-off, tourist trap"],
    "Presupuesto y gasto": ["presupuesto del viaje, gasto, ahorrar dinero, comida y hospedaje económicos",
                            "trip budget, spending, saving money"],
}

# Palabras clave (validación léxica independiente de la similitud semántica).
_LEX_PRECIO = ("precio precios costo costos costar cuesta cuestan caro caros barato baratos economico "
               "gratis gratuito tarifa cobrar cobran pagar pago dinero pesos price prices cost costs "
               "expensive cheap affordable value worth money fee fees pay paid free preco custo prix "
               "cout cher preis kosten teuer prezzo")


def _unit(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _tokens_ancla(txt):
    """Tokens de una frase ancla en la misma forma que 'tokens_limpios' (sin acentos y lematizados
    en cada idioma posible)."""
    sal = set()
    for t in limpiar_texto(txt)[0]:
        sal.add(sin_acentos(t))
        for lg in ("es", "en", "pt", "fr", "de", "it"):
            sal.add(sin_acentos(lema(t, lg)))
    return sal


def _wilson(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _mwu_cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p), float(2 * u / (len(a) * len(b)) - 1)


def _espacio_precio(datos, A):
    """Devuelve (E, c, F, nombre_espacio, con_embeddings): comentarios, concepto y facetas en
    el mismo espacio vectorial (filas de norma 1). Por omisión LSA sobre el texto limpio (sin
    descargar modelos); con --similitud-embeddings usa embeddings multilingües."""
    if getattr(A, "similitud_embeddings", False):
        textos = [_texto_para_bert(t) for t in datos["texto_original"]]
        try:
            from sentence_transformers import SentenceTransformer
            modelo = SentenceTransformer(A.bt_modelo)

            def enc(x):
                return normalize(np.asarray(modelo.encode(list(x), batch_size=64,
                                                          show_progress_bar=False), dtype=float))
            E = enc(textos)
            c = _unit(enc(ANCLAS_PRECIO).mean(axis=0))
            F = np.vstack([_unit(enc(v).mean(axis=0)) for v in FACETAS_PRECIO.values()])
            return E, c, F, f"embeddings multilingües ({A.bt_modelo})", True
        except Exception as e:
            print(f"AVISO: no se pudieron usar embeddings ({type(e).__name__}). Se usa LSA.")
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, max_df=.95,
                          min_df=2 if len(datos) >= 300 else 1)
    X = vec.fit_transform(datos["texto_limpio"])
    k = min(100, X.shape[1] - 1, X.shape[0] - 1)
    if k < 2:
        raise ValueError("vocabulario insuficiente para LSA")
    svd = TruncatedSVD(n_components=k, random_state=A.semilla).fit(X)
    E = normalize(svd.transform(X))

    def proy(frases):
        toks = set().union(*[_tokens_ancla(f) for f in frases])
        return _unit(svd.transform(vec.transform([" ".join(sorted(toks))])).ravel())
    c = proy(ANCLAS_PRECIO)
    F = np.vstack([proy(v) for v in FACETAS_PRECIO.values()])
    if not c.any():
        raise ValueError("ninguna palabra del concepto aparece en el vocabulario")
    return E, c, F, "LSA (léxico-semántico, sin embeddings)", False


def _terminos_contexto(tok_a, tok_b, top=12, min_cnt=3):
    """Palabras más frecuentes (en % de comentarios) y más distintivas (log-odds con prior
    informativo, z) entre dos conjuntos de comentarios."""
    ca, cb = Counter(), Counter()
    for t in tok_a:
        ca.update(set(t))
    for t in tok_b:
        cb.update(set(t))
    na_c, nb_c = max(len(tok_a), 1), max(len(tok_b), 1)
    frec = pd.DataFrame(
        [{"grupo": G_DMO, "termino": w, "n": n, "pct_comentarios": 100 * n / na_c}
         for w, n in ca.most_common(top)] +
        [{"grupo": G_PERC, "termino": w, "n": n, "pct_comentarios": 100 * n / nb_c}
         for w, n in cb.most_common(top)])
    na, nb = sum(ca.values()), sum(cb.values())
    a0 = 0.1 * (na + nb)
    filas = []
    for w in (set(ca) | set(cb)):
        if ca[w] + cb[w] < min_cnt or len(w) < 2:
            continue
        aw = 0.1 * (ca[w] + cb[w])
        la = np.log((ca[w] + aw) / max(na + a0 - ca[w] - aw, 1e-9))
        lb = np.log((cb[w] + aw) / max(nb + a0 - cb[w] - aw, 1e-9))
        z = (la - lb) / np.sqrt(1 / (ca[w] + aw) + 1 / (cb[w] + aw))
        filas.append({"termino": w, "n_DMO": ca[w], "n_percibida": cb[w], "z": float(z)})
    dist = pd.DataFrame(filas)
    if len(dist):
        dist = dist.sort_values("z", ascending=False)
        dist = pd.concat([dist.head(top).assign(distintivo_de=G_DMO),
                          dist.tail(top).iloc[::-1].assign(distintivo_de=G_PERC)])
    return frec, dist


def _tooltip_precio(r):
    pol = "—" if not np.isfinite(r.polaridad) else f"{r.polaridad:+.2f}"
    txt = f"{r.fuente} · {r.grupo}\nsimilitud = {r.sim_precio:.3f} · polaridad = {pol}"
    if r.faceta:
        txt += f"\nfaceta: {r.faceta}"
    return txt + "\n" + textwrap.fill(recortar(r.texto_original, 260), 60)


def _dibujar_fuentes(ax, sub, xcol, ycol, estilos, orden, tam_col=None):
    """Dibuja un scatter por fuente (relacionados resaltados con borde negro). Devuelve
    (colecciones, grupos): colecciones = [(PathCollection, tooltips)], grupos = {fuente: [PathCollection]}."""
    colecciones, grupos = [], {}
    for f in orden:
        col, mk = estilos[f]
        grupos[f] = []
        for rel, alfa in ((False, .2), (True, .9)):
            s = sub[(sub["fuente"] == f) & (sub["relacionado"] == rel)]
            if s.empty:
                continue
            tam = 10 if not rel else (s[tam_col].to_numpy() if tam_col else 40)
            sc = ax.scatter(s[xcol], s[ycol], s=tam, c=col, marker=mk, alpha=alfa,
                            edgecolors="black" if rel else "none", linewidths=.8,
                            zorder=3 if rel else 2)
            colecciones.append((sc, [_tooltip_precio(r) for r in s.itertuples()]))
            grupos[f].append(sc)
    return colecciones, grupos


def _activar_interaccion(fig, ax, colecciones, grupos, estilos, extras=()):
    """Tooltip con el texto del comentario al pasar el cursor y clic en la leyenda para
    mostrar/ocultar una fuente (zoom y desplazamiento: barra de herramientas de matplotlib)."""
    from matplotlib.lines import Line2D
    anot = ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points", fontsize=8,
                       bbox=dict(boxstyle="round", fc="white", ec="black", alpha=.96),
                       arrowprops=dict(arrowstyle="->"), zorder=100)
    anot.set_visible(False)

    def al_mover(ev):
        if ev.inaxes is not ax:
            if anot.get_visible():
                anot.set_visible(False)
                fig.canvas.draw_idle()
            return
        for sc, textos in colecciones:
            if not sc.get_visible():
                continue
            hit, info = sc.contains(ev)
            if hit and len(info["ind"]):
                i = int(info["ind"][0])
                anot.xy = sc.get_offsets()[i]
                anot.set_text(textos[i])
                anot.set_visible(True)
                fig.canvas.draw_idle()
                return
        if anot.get_visible():
            anot.set_visible(False)
            fig.canvas.draw_idle()

    nombres = list(estilos)
    hs = [Line2D([], [], ls="", marker=mk, color=col, markeredgecolor="black", markersize=8,
                 label=nom) for nom, (col, mk) in estilos.items()] + list(extras)
    leg = ax.legend(handles=hs, title="Fuente (clic = ocultar/mostrar)", loc="best", fontsize=8)
    lhs = getattr(leg, "legend_handles", None) or leg.legendHandles
    mapa = {}
    for lh, nom in zip(lhs, nombres):
        lh.set_picker(True)
        lh.set_pickradius(8)
        mapa[lh] = nom

    def al_clic(ev):
        nom = mapa.get(ev.artist)
        if nom is None or not grupos.get(nom):
            return
        vis = not grupos[nom][0].get_visible()
        for sc in grupos[nom]:
            sc.set_visible(vis)
        ev.artist.set_alpha(1.0 if vis else .25)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", al_mover)
    fig.canvas.mpl_connect("pick_event", al_clic)
    fig._callbacks_precio = (al_mover, al_clic)      # referencia fuerte: si no, matplotlib los descarta


def _scatter_interactivos_precio(datos, E, c, A, umbral, orden, nombre_esp, top_idx):
    """Dos scatter interactivos de matplotlib (sin web ni plotly): (A) mapa semántico PCA con
    el concepto como estrella; (B) similitud con el concepto vs. polaridad. Devuelve [(nombre, fig)]."""
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA

    n = len(datos)
    rel_all = datos["relacionado"].to_numpy(bool)
    pca = PCA(n_components=2, random_state=A.semilla)
    Z = pca.fit_transform(E)
    zc = pca.transform(c[None, :])[0]
    ev = pca.explained_variance_ratio_

    # Muestreo (el hover se vuelve lento con decenas de miles de puntos): se conservan los relacionados.
    rng = np.random.default_rng(A.semilla)
    keep = np.arange(n)
    if A.max_puntos_precio and n > A.max_puntos_precio:
        must, resto = np.where(rel_all)[0], np.where(~rel_all)[0]
        if len(must) >= A.max_puntos_precio:
            keep = rng.choice(must, A.max_puntos_precio, replace=False)
        else:
            keep = np.concatenate([must, rng.choice(resto, min(A.max_puntos_precio - len(must),
                                                               len(resto)), replace=False)])
        keep = np.union1d(keep, top_idx)
        print(f"  (scatter: se dibujan {len(keep):,} de {n:,} comentarios; --max-puntos-precio 0 los dibuja todos)")
    sub = datos.iloc[keep].copy()
    sub["x_map"], sub["y_map"] = Z[keep, 0], Z[keep, 1]
    smin, smax = float(datos["sim_precio"].min()), float(datos["sim_precio"].max())
    sub["tam"] = 25 + 170 * np.clip((sub["sim_precio"].to_numpy() - smin) / max(smax - smin, 1e-9), 0, 1)

    marcas = [m for m in MARCAS if m != "D"]
    estilos, i_otra = {}, 0
    for f in orden:
        if f == str(A.dmo):
            estilos[f] = (C_DMO, "D")
        else:
            estilos[f] = (C_FUENTES[i_otra % len(C_FUENTES)], marcas[i_otra % len(marcas)])
            i_otra += 1
    top = datos.loc[top_idx]
    extras = [Line2D([], [], ls="", marker="*", color="gold", markeredgecolor="black", markersize=14,
                     label='Concepto "precio / valor / costo"'),
              Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor="black",
                     markersize=11, label=f"Top {len(top_idx)} más cercanos")]

    # ---------- A) mapa semántico ----------
    figA, ax = plt.subplots(figsize=(12, 8.2))
    colecciones, grupos = _dibujar_fuentes(ax, sub, "x_map", "y_map", estilos, orden, tam_col="tam")
    sc_top = ax.scatter(Z[top_idx, 0], Z[top_idx, 1], s=360, facecolors="none", edgecolors="black",
                        linewidths=2, zorder=5)
    for i, (xx, yy) in enumerate(zip(Z[top_idx, 0], Z[top_idx, 1]), 1):
        ax.annotate(str(i), (xx, yy), xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", zorder=6)
    colecciones.insert(0, (sc_top, [_tooltip_precio(r) for r in top.itertuples()]))
    ax.scatter([zc[0]], [zc[1]], s=650, marker="*", c="gold", edgecolors="black", linewidths=1.5, zorder=7)
    ax.annotate("Concepto", (zc[0], zc[1]), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", zorder=7)
    ax.set_xlabel(f"Componente principal 1 ({ev[0]:.0%} de la varianza)")
    ax.set_ylabel(f"Componente principal 2 ({ev[1]:.0%} de la varianza)")
    ax.set_title('Comentarios vs. concepto "precio / valor / costo" — mapa semántico\n'
                 f"{nombre_esp} · tamaño = similitud con el concepto · borde negro = relacionado "
                 f"(coseno ≥ {umbral:.2f}) · forma/color = fuente", fontsize=10)
    ax.grid(alpha=.2)
    _activar_interaccion(figA, ax, colecciones, grupos, estilos, extras)

    # ---------- B) similitud vs. polaridad ----------
    figB, ax2 = plt.subplots(figsize=(11, 7.2))
    sb = sub[np.isfinite(sub["polaridad"].to_numpy(float))]
    colecciones2, grupos2 = _dibujar_fuentes(ax2, sb, "sim_precio", "polaridad", estilos, orden)
    ax2.axvline(umbral, color="black", ls="--", lw=1)
    ax2.text(umbral, 1.0, " umbral de relación", transform=ax2.get_xaxis_transform(), va="top", fontsize=8)
    ax2.axhline(A.umbral, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Similitud coseno con el concepto")
    ax2.set_ylabel("Polaridad (−1 negativo … +1 positivo)")
    ax2.set_title('Similitud con "precio / valor / costo" vs. tono de cada comentario\n'
                  "a la derecha de la línea punteada: comentarios relacionados con el concepto", fontsize=10)
    ax2.grid(alpha=.2)
    _activar_interaccion(figB, ax2, colecciones2, grupos2, estilos)
    return [("13a_precio_mapa_semantico", figA), ("13b_precio_similitud_vs_tono", figB)]


def analizar_precio_valor_costo(normales, A):
    """Análisis específico del concepto sintético "precio / valor / costo"."""
    titulo('ANÁLISIS ESPECÍFICO DEL CONCEPTO "PRECIO / VALOR / COSTO"')
    datos = normales.copy()
    datos["texto_limpio"] = datos["texto_limpio"].fillna("").astype(str)
    datos = datos[datos["texto_limpio"].str.strip().ne("")].reset_index(drop=True)
    datos["fuente"] = datos["fuente"].astype(str)
    es_dmo = datos["es_dmo"].to_numpy(bool)
    if es_dmo.sum() < 2 or (~es_dmo).sum() < 2:
        print("No se puede analizar: se necesitan al menos 2 comentarios en la DMO y 2 en la percibida.")
        return
    try:
        E, c, F, nombre_esp, con_emb = _espacio_precio(datos, A)
    except Exception as e:
        print(f"No se pudo construir el concepto ({type(e).__name__}: {e}).")
        return

    # ---- 1. Similitud semántica con el concepto ------------------------------------
    lex = _tokens_ancla(_LEX_PRECIO)
    lx = np.array([bool(lex.intersection(t)) for t in datos["tokens_limpios"]], dtype=bool)
    n_semilla = 0
    if not con_emb and lx.sum() >= 5:
        # Expansión por retroalimentación (Rocchio): sin modelos externos, el concepto se enriquece con
        # el contexto de los comentarios que ya usan palabras de precio (captura sinónimos del corpus).
        c = _unit(.5 * c + .5 * _unit(E[lx].mean(axis=0)))
        n_semilla = int(lx.sum())
    sim = E @ c
    if A.umbral_precio is not None:
        umbral, modo = float(A.umbral_precio), "fijado con --umbral-precio"
    else:
        piso = .30 if con_emb else 0.0
        umbral = max(float(sim.mean() + A.z_precio * sim.std(ddof=1)), piso)
        modo = f"automático: media + {A.z_precio:g} desv. est. de las similitudes (piso {piso:.2f})"
    rel = sim >= umbral
    datos["sim_precio"] = sim
    datos["relacionado"] = rel
    datos["grupo"] = np.where(es_dmo, G_DMO, G_PERC)
    nombres_f = list(FACETAS_PRECIO)
    datos["faceta"] = np.where(rel, np.array(nombres_f)[np.argmax(E @ F.T, axis=1)], "")
    datos["menciona_lexico"] = lx
    pol = datos["polaridad"].to_numpy(float)
    orden = _orden_fuentes(datos["fuente"].unique(), A)

    print(f"\nEspacio vectorial : {nombre_esp}")
    print(f"Concepto          : vector medio de {len(ANCLAS_PRECIO)} frases ancla en 6 idiomas"
          + (f" + expansión por retroalimentación con {n_semilla} comentarios que usan palabras de precio"
             if n_semilla else ""))
    print(f"Umbral de relación: coseno ≥ {umbral:.3f} ({modo})")
    print(f"Comentarios relacionados: {int(rel.sum()):,} de {len(datos):,} "
          f"({100 * rel.mean():.1f}%)")
    lx = datos["menciona_lexico"].to_numpy(bool)
    if rel.sum() and lx.sum():
        print(f"Validación léxica : {100 * lx[rel].mean():.0f}% de los relacionados contiene alguna palabra "
              f"clave de precio; {100 * rel[lx].mean():.0f}% de los comentarios con palabra clave fueron "
              "detectados semánticamente.")
    if not rel.any():
        print("Ningún comentario supera el umbral: baja --umbral-precio o --z-precio.")

    # ---- 2. Frecuencia: DMO vs percibida ---------------------------------------------
    titulo("FRECUENCIA DEL CONCEPTO: IMAGEN PROYECTADA (DMO) VS PERCIBIDA", nivel=2)
    filas = []

    def fila(nivel, nombre, m):
        n, k = int(m.sum()), int((rel & m).sum())
        lo, hi = _wilson(k, n)
        filas.append({"nivel": nivel, "nombre": nombre, "n_comentarios": n, "n_relacionados": k,
                      "pct_relacionados": 100 * k / n if n else np.nan, "IC95_inf": 100 * lo,
                      "IC95_sup": 100 * hi, "sim_media": float(sim[m].mean()) if n else np.nan,
                      "pct_menciona_lexico": 100 * float(lx[m].mean()) if n else np.nan})
    fila("grupo", G_DMO, es_dmo)
    fila("grupo", G_PERC, ~es_dmo)
    for f in orden:
        fila("fuente", f, (datos["fuente"] == f).to_numpy())
    frec = pd.DataFrame(filas)
    print(frec.round(2).to_string(index=False))
    ak, an = int((rel & es_dmo).sum()), int(es_dmo.sum())
    bk, bn = int((rel & ~es_dmo).sum()), int((~es_dmo).sum())
    pf = {}
    if (ak + bk) > 0 and (an - ak + bn - bk) > 0:
        chi2, p_chi, _, _ = chi2_contingency([[ak, an - ak], [bk, bn - bk]])
        orr = ((ak + .5) * (bn - bk + .5)) / ((an - ak + .5) * (bk + .5))
        pf.update(chi2=chi2, p=p_chi, phi=float(np.sqrt(chi2 / (an + bn))), OR=orr)
        print(f"\nχ² (DMO vs percibida, % relacionados): χ²={chi2:.2f}, p={p_chi:.4g}, "
              f"φ={pf['phi']:.3f}; razón de momios DMO/percibida = {orr:.2f}")
    u, p_u, d = _mwu_cliff(sim[es_dmo], sim[~es_dmo])
    if np.isfinite(u):
        print(f"Mann-Whitney sobre la similitud con el concepto: p={p_u:.4g}, δ de Cliff={d:+.3f} "
              f"({_magnitud_cliff(d)}; δ>0 = la DMO está más cerca del concepto)")
    _resumen("Precio/valor/costo", "similitud con el concepto", G_DMO, sim[es_dmo])
    _resumen("Precio/valor/costo", "similitud con el concepto", G_PERC, sim[~es_dmo])

    # ---- 3. Tono ----------------------------------------------------------------------
    titulo("TONO CON EL QUE SE ABORDA EL CONCEPTO (polaridad −1…+1)", nivel=2)
    filas_t = []
    for gnom, mg in ((G_DMO, es_dmo), (G_PERC, ~es_dmo)):
        for sub_nom, mr in (("relacionados con precio", rel), ("resto de comentarios", ~rel)):
            x = pol[mg & mr]
            x = x[np.isfinite(x)]
            filas_t.append({"grupo": gnom, "subconjunto": sub_nom, **_desc(x),
                            "pct_positivo": 100 * float(np.mean(x >= A.umbral)) if len(x) else np.nan})
    tono = pd.DataFrame(filas_t)
    print(tono.round(3).to_string(index=False))
    u, p_u, d = _mwu_cliff(pol[es_dmo & rel], pol[~es_dmo & rel])
    if np.isfinite(u):
        print(f"\nTono en comentarios relacionados, DMO vs percibida: Mann-Whitney p={p_u:.4g}, "
              f"δ de Cliff={d:+.3f} ({_magnitud_cliff(d)}; δ>0 = DMO más positiva)")
    for gnom, mg in ((G_DMO, es_dmo), (G_PERC, ~es_dmo)):
        u2, p2, d2 = _mwu_cliff(pol[mg & rel], pol[mg & ~rel])
        if np.isfinite(u2):
            print(f"  {gnom}: relacionados vs resto: p={p2:.4g}, δ={d2:+.3f} ({_magnitud_cliff(d2)}; "
                  "δ<0 = hablar de precio es más negativo que el resto)")
        _resumen("Precio/valor/costo", "polaridad de comentarios relacionados", gnom, pol[mg & rel])
    print("\n(Se usa la polaridad numérica real; la etiqueta 'positiva por diseño' de la DMO no interviene.)")

    # ---- 4. Contexto: facetas y términos --------------------------------------------
    titulo("CONTEXTO: DE QUÉ FORMA SE HABLA DEL CONCEPTO", nivel=2)
    cont = (pd.crosstab(datos.loc[rel, "faceta"], datos.loc[rel, "grupo"])
            .reindex(index=nombres_f, columns=[G_DMO, G_PERC], fill_value=0))
    fac = pd.DataFrame({"faceta": nombres_f})
    for g in (G_DMO, G_PERC):
        k_ = "DMO" if g == G_DMO else "percibida"
        tot = max(int(cont[g].sum()), 1)
        fac[f"n_{k_}"] = cont[g].to_numpy()
        fac[f"pct_{k_}"] = 100 * cont[g].to_numpy() / tot
        fac[f"polaridad_{k_}"] = [
            float(np.nanmean(pol[(datos["faceta"] == fn).to_numpy() & (datos["grupo"] == g).to_numpy()])
                  ) if cont.loc[fn, g] and np.isfinite(
                pol[(datos["faceta"] == fn).to_numpy() & (datos["grupo"] == g).to_numpy()]).any()
            else np.nan for fn in nombres_f]
    print("Facetas (cada comentario relacionado se asigna a la faceta más cercana):")
    print(fac.round(2).to_string(index=False))
    if (cont.sum(axis=0) > 0).all():
        cc = cont.loc[cont.sum(axis=1) > 0]
        if cc.shape[0] >= 2:
            chi2f, pf_, _, _ = chi2_contingency(cc.to_numpy())
            pa = cont[G_DMO].to_numpy() / cont[G_DMO].sum()
            pb = cont[G_PERC].to_numpy() / cont[G_PERC].sum()
            js = float(jensenshannon(pa, pb, base=2))
            print(f"\nχ² de las facetas DMO vs percibida: p={pf_:.4g} | divergencia de Jensen-Shannon "
                  f"= {js:.3f} (0 = misma distribución, 1 = totalmente distinta)")

    tok_d = datos.loc[rel & es_dmo, "tokens_limpios"].tolist()
    tok_p = datos.loc[rel & ~es_dmo, "tokens_limpios"].tolist()
    terminos, distintivos = pd.DataFrame(), pd.DataFrame()
    if tok_d and tok_p:
        terminos, distintivos = _terminos_contexto(tok_d, tok_p)
        print("\nPalabras más frecuentes en los comentarios relacionados (% de comentarios del grupo):")
        for g in (G_DMO, G_PERC):
            s = terminos[terminos["grupo"] == g]
            print(f"  {g:18s}: " + ", ".join(f"{r.termino} ({r.pct_comentarios:.0f}%)"
                                             for r in s.itertuples()))
        if len(distintivos):
            print("\nPalabras más distintivas (z de log-odds; mayor |z| = más exclusiva del grupo):")
            for g in (G_DMO, G_PERC):
                s = distintivos[distintivos["distintivo_de"] == g]
                print(f"  {g:18s}: " + ", ".join(f"{r.termino} ({r.z:+.1f})" for r in s.itertuples()))
    else:
        print("\nNo hay comentarios relacionados en ambos grupos para comparar términos.")

    # ---- 5. Comentarios más cercanos ---------------------------------------------------
    titulo(f"LOS {A.top_precio} COMENTARIOS MÁS CERCANOS AL CONCEPTO", nivel=2)
    cols = ["fuente", "grupo", "sim_precio", "polaridad", "faceta", "texto_original"]
    top_all = datos.sort_values("sim_precio", ascending=False).head(A.top_precio)
    tops = [("TODOS LOS COMENTARIOS", top_all)]
    tops += [(g, datos[datos["grupo"] == g].sort_values("sim_precio", ascending=False)
              .head(A.top_precio)) for g in (G_DMO, G_PERC)]
    filas_top = []
    for nombre, t in tops:
        print(f"\n[{nombre}]")
        for i, r in enumerate(t[cols].itertuples(), 1):
            pol_s = "—" if not np.isfinite(r.polaridad) else f"{r.polaridad:+.2f}"
            print(f" {i}. fuente={r.fuente} | {r.grupo} | similitud={r.sim_precio:.3f} | "
                  f"polaridad={pol_s}\n    «{recortar(r.texto_original, 260)}»")
            filas_top.append({"ranking_en": nombre, "posicion": i, **r._asdict()})
    top_idx = top_all.index.to_numpy()

    # ---- 6. Resumen interpretativo -------------------------------------------------------
    pa_, pb_ = 100 * ak / max(an, 1), 100 * bk / max(bn, 1)
    ma = np.nanmean(pol[es_dmo & rel]) if np.isfinite(pol[es_dmo & rel]).any() else np.nan
    mb = np.nanmean(pol[~es_dmo & rel]) if np.isfinite(pol[~es_dmo & rel]).any() else np.nan
    print("\nLECTURA RÁPIDA")
    print(f"  Frecuencia: el concepto aparece en {pa_:.1f}% de los comentarios de la DMO y en {pb_:.1f}% "
          f"de la imagen percibida ({'la DMO lo enfatiza más' if pa_ > pb_ else 'la imagen percibida lo menciona más'}).")
    if np.isfinite(ma) and np.isfinite(mb):
        print(f"  Tono: polaridad media {ma:+.2f} en la DMO vs {mb:+.2f} en la percibida "
              f"({'la DMO lo presenta más favorablemente' if ma > mb else 'la percibida lo valora mejor'}).")

    # ---- 7. Gráficas (matplotlib; los scatter son interactivos al mostrarse con plt.show()) ----
    for nombre, fig in _scatter_interactivos_precio(datos, E, c, A, umbral, orden, nombre_esp, top_idx):
        _reg(nombre, fig)

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8))
    ax = axs[0]
    etiquetas, i_otra = [], 0
    for j, r in frec.iterrows():
        if r["nombre"] == G_DMO or (r["nivel"] == "fuente" and r["nombre"] == str(A.dmo)):
            col, h = C_DMO, H_DMO
        elif r["nombre"] == G_PERC:
            col, h = C_PERC, H_PERC
        else:
            col, h = C_FUENTES[i_otra % len(C_FUENTES)], HATCHES[i_otra % len(HATCHES)]
            i_otra += 1
        ax.bar(j, r["pct_relacionados"], .7, color=col, edgecolor="black", hatch=h,
               yerr=[[r["pct_relacionados"] - r["IC95_inf"]], [r["IC95_sup"] - r["pct_relacionados"]]], capsize=4)
        ax.text(j, r["IC95_sup"] + .3, f"{r['pct_relacionados']:.1f}%", ha="center", fontsize=8)
        etiquetas.append(("\n".join(textwrap.wrap(str(r["nombre"]), 12))))
    ax.set_xticks(range(len(frec)))
    ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_ylabel("% de comentarios relacionados (IC 95% de Wilson)")
    ax.set_title("Frecuencia del concepto")
    ax.grid(axis="y", alpha=.25)

    ax = axs[1]
    w = .38
    for gi, (gnom, col, h) in enumerate(((G_DMO, C_DMO, H_DMO), (G_PERC, C_PERC, H_PERC))):
        for k_, (sub_nom, alpha, hh) in enumerate((("relacionados con precio", .95, h), ("resto de comentarios", .35, ""))):
            fila_t = tono[(tono["grupo"] == gnom) & (tono["subconjunto"] == sub_nom)].iloc[0]
            v = fila_t["media"]
            xpos = gi + (k_ - .5) * w
            if np.isfinite(v):
                ax.bar(xpos, v, w, color=col, alpha=alpha, edgecolor="black", hatch=hh)
                ax.text(xpos, v + (.01 if v >= 0 else -.01), f"{v:+.2f}\n(n={int(fila_t['n'])})",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=.8)
    ax.margins(y=.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([G_DMO, G_PERC])
    ax.set_ylabel("Polaridad media")
    ax.set_title("Tono: comentarios de precio (trama) vs resto (claro)")
    ax.grid(axis="y", alpha=.25)

    ax = axs[2]
    x = np.arange(len(nombres_f))
    b1 = ax.bar(x - .2, fac["pct_DMO"], .4, color=C_DMO, edgecolor="black", hatch=H_DMO, label=G_DMO)
    b2 = ax.bar(x + .2, fac["pct_percibida"], .4, color=C_PERC, edgecolor="black", hatch=H_PERC, label=G_PERC)
    ax.bar_label(b1, fmt="%.0f%%", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.0f%%", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["\n".join(textwrap.wrap(n_, 11)) for n_ in nombres_f], fontsize=8)
    ax.set_ylabel("% de los comentarios relacionados del grupo")
    ax.set_title("Contexto: facetas del concepto")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.suptitle('Concepto "precio / valor / costo": imagen proyectada (DMO) vs percibida', fontsize=13)
    _reg("13c_precio_valor_costo_comparacion", fig)

    # ---- 8. Resultados para --guardar -------------------------------------------------------
    A.precio_frecuencia = frec
    A.precio_tono = tono
    A.precio_facetas = fac
    A.precio_terminos = pd.concat([terminos.assign(tipo="frecuentes"),
                                   distintivos.assign(tipo="distintivos")], ignore_index=True) \
        if len(terminos) else pd.DataFrame()
    A.precio_top = pd.DataFrame(filas_top).drop(columns=["Index"], errors="ignore")
    A.precio_comentarios = datos[["fuente", "grupo", "sim_precio", "relacionado", "menciona_lexico",
                                  "polaridad", "faceta", "texto_original"]]


# ------------------------------------------------------------
# 7.4 RESUMEN ESTADÍSTICO DE TODAS LAS COMPARACIONES
# ------------------------------------------------------------

def resumen_estadistico(A):
    titulo("RESUMEN ESTADÍSTICO DE LAS COMPARACIONES (media, mediana, desviación estándar)")
    if not RESUMEN:
        print("No hay comparaciones calculadas.")
        return
    tabla = pd.DataFrame(RESUMEN)
    A.resumen_estadistico = tabla
    for comp, sub in tabla.groupby("comparacion", sort=False):
        print(f"\n--- {comp} ---")
        print(sub.drop(columns="comparacion").round(4).to_string(index=False))
    print("\nNotas: desviación estándar muestral (n-1). En las comparaciones de tópicos, la media y "
          "la desviación son ENTRE tópicos (cuánto varía el peso de un tópico a otro). En similitud "
          "bootstrap son la media y desviación de las réplicas.")


def argumentos():
    p = crear_parser()
    p.description = "Pipeline completo DMO vs imagen percibida (limpieza, atípicos, sentimiento, tópicos)"
    p.add_argument("--n-topicos", type=int, default=None,
                   help="número fijo de tópicos (por omisión se elige por coherencia NPMI)")
    p.add_argument("--min-topicos", type=int, default=3, help="mínimo de k al buscar")
    p.add_argument("--max-topicos", type=int, default=8, help="máximo de k al buscar")
    p.add_argument("--bertopic", action="store_true", help="usar BERTopic (opcional)")
    p.add_argument("--bt-modelo", default="paraphrase-multilingual-MiniLM-L12-v2",
                   help="modelo de sentence-transformers para BERTopic")
    p.add_argument("--bt-min-tam", type=int, default=8, help="tamaño mínimo de tópico en BERTopic")
    p.add_argument("--similitud-embeddings", action="store_true",
                   help="añade embeddings multilingües (sentence-transformers) a la similitud")
    p.add_argument("--n-boot", type=int, default=200,
                   help="réplicas bootstrap para el índice de alineación (similitud)")
    p.add_argument("--sin-precio", action="store_true",
                   help="omite el análisis del concepto 'precio / valor / costo'")
    p.add_argument("--umbral-precio", type=float, default=None,
                   help="coseno mínimo con el concepto para considerar un comentario relacionado "
                        "(por omisión: media + z desv. est., con piso 0.30 si hay embeddings)")
    p.add_argument("--z-precio", type=float, default=1.5,
                   help="z del umbral automático (media + z·desv. est. de las similitudes)")
    p.add_argument("--top-precio", type=int, default=5,
                   help="cuántos comentarios más cercanos al concepto se muestran")
    p.add_argument("--max-puntos-precio", type=int, default=20000,
                   help="máximo de puntos en el scatter interactivo (0 = todos)")
    p.add_argument("--ocultar-graficas-previas", action="store_true",
                   help="no mostrar las gráficas de limpieza, atípicos y n-gramas")
    p.add_argument("--paleta", default="cividis", choices=PALETAS_CONTINUAS,
                   help="paleta continua de los mapas de calor: viridis, plasma, inferno, "
                        "magma o cividis (por defecto cividis)")
    A = p.parse_args()
    global CMAP
    CMAP = A.paleta
    return A


def main():
    A = argumentos()
    df = ejecutar_base(A)
    graficar_resumen_limpieza(df)
    df = detectar_isolation_forest_vectorial(df, A)
    mostrar_ngramas(df)
    mostrar_atipicos_y_nubes(df, A)
    n_total, n_out = len(df), int(df["outlier"].sum())
    print(f"\nResumen: {n_total - n_out:,} comentarios normales y {n_out:,} atípicos "
          f"excluidos del análisis posterior.")
    if A.ocultar_graficas_previas:
        plt.close("all")

    normales = df.loc[~df["outlier"]].reset_index(drop=True)
    titulo("SENTIMIENTO HÍBRIDO (BERT + LÉXICO) SOBRE COMENTARIOS NORMALES")
    print(f"Se analizan {len(normales):,} comentarios normales.")
    normales = clasificar_hibrido(normales, A)
    pos, neg = separar_particiones(normales)
    imprimir_reporte(normales, pos, neg, A, n_total, n_out)
    metodo = "BERT + léxico" if normales["pol_bert"].notna().any() else "solo léxico"
    FIGS.clear()
    g_distribucion_fuentes(normales, A, metodo)
    g_polaridad(normales, A, metodo)
    g_bert_vs_lexico(normales, A)
    g_validacion(normales)
    g_terminos(pos, neg)
    g_fuente_por_grupo(pos, neg, A)
    g_nubes(pos, neg, A)

    # Comparación independiente: todos los comentarios normales, sin separar
    # por sentimiento, usando un vocabulario y un NMF compartidos.
    comparar_topicos_dmo_percibida(normales, A)
    comparar_sentimiento(normales, A)
    comparar_proporciones_sentimiento(normales, A)
    comparar_similitud(normales, A)
    if not A.sin_precio:
        analizar_precio_valor_costo(normales, A)

    titulo("MODELADO DE TÓPICOS POR SENTIMIENTO")
    g_decision(pos, neg, A)
    pos = procesar_particion(pos, "positiva", C_POS, H_POS, "pos", A, es_negativa=False)
    neg = procesar_particion(neg, "negativa", C_NEG, H_NEG, "neg", A, es_negativa=True)

    comparar_topicos_buenos_malos(pos, neg, A)
    resumen_estadistico(A)

    if A.guardar:
        os.makedirs(A.guardar, exist_ok=True)
        for nombre, fig in FIGS:
            fig.savefig(os.path.join(A.guardar, nombre + ".png"), dpi=150, bbox_inches="tight")
        cols = ["fuente", "idioma", "texto_original", "polaridad", "etiqueta", "topico"]
        pd.concat([pos[cols], neg[cols]]).to_csv(
            os.path.join(A.guardar, "comentarios_por_topico.csv"), index=False, encoding="utf-8-sig")
        filas = []
        for nombre, g in (("positiva", pos), ("negativa", neg)):
            for t, terms in enumerate(g.attrs.get("terminos", []), 1):
                filas.append({"particion": nombre, "topico": t,
                              "n": int((g["topico"] == t).sum()),
                              "terminos": ", ".join(w for w, _ in terms)})
        pd.DataFrame(filas).to_csv(os.path.join(A.guardar, "terminos_por_topico.csv"),
                                   index=False, encoding="utf-8-sig")
        for nom, g in (("particion_positiva", pos), ("particion_negativa", neg)):
            g[COLS_SALIDA + ["texto_limpio", "topico"]].to_csv(
                os.path.join(A.guardar, nom + ".csv"), index=False, encoding="utf-8-sig")
        if hasattr(A, "comparacion_topicos"):
            A.comparacion_topicos.to_csv(
                os.path.join(A.guardar, "comparacion_topicos_DMO_percibida.csv"),
                index=False, encoding="utf-8-sig")
            A.topicos_por_fuente.to_csv(
                os.path.join(A.guardar, "topicos_por_fuente.csv"),
                encoding="utf-8-sig")
            A.comentarios_topicos_comparacion.to_csv(
                os.path.join(A.guardar, "comentarios_topicos_comparacion.csv"),
                index=False, encoding="utf-8-sig")
        extras = {
            "sentimiento_por_fuente": "sentimiento_por_fuente",
            "sentimiento_pruebas": "sentimiento_pruebas_estadisticas",
        "prop_sent_global": "proporciones_sentimiento_global",
        "prop_sent_pruebas_global": "proporciones_sentimiento_pruebas_global",
        "prop_sent_descriptivos_topicos": "proporciones_sentimiento_descriptivos_por_topico",
        "prop_sent_topicos": "proporciones_sentimiento_pruebas_por_topico",
            "similitud_global": "similitud_global",
            "similitud_por_fuente": "similitud_por_fuente",
            "comparacion_topicos_positivos": "comparacion_topicos_buenos",
            "comparacion_topicos_negativos": "comparacion_topicos_malos",
            "resumen_estadistico": "resumen_estadistico_comparaciones",
            "precio_frecuencia": "precio_frecuencia_por_grupo",
            "precio_tono": "precio_tono",
            "precio_facetas": "precio_facetas",
            "precio_terminos": "precio_terminos_contexto",
            "precio_top": "precio_top_comentarios",
            "precio_comentarios": "precio_similitud_todos_los_comentarios",
        }
        for atributo, archivo in extras.items():
            if hasattr(A, atributo):
                getattr(A, atributo).to_csv(os.path.join(A.guardar, archivo + ".csv"),
                                            index=False, encoding="utf-8-sig")
        for nom, M in getattr(A, "similitud_matrices", {}).items():
            seguro = re.sub(r"\W+", "_", sin_acentos(nom)).strip("_").lower()
            M.to_csv(os.path.join(A.guardar, f"similitud_matriz_{seguro}.csv"),
                     encoding="utf-8-sig")
        print(f"\nGuardado en '{A.guardar}': {len(FIGS)} PNG + CSV.")

    if A.sin_graficas:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
