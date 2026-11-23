# Tablero de PLN: imagen proyectada (DMO) vs imagen percibida.
#
# Dos archivos:
#   streamlit run tablero.py -- dmo.csv comparar.csv ColumnaComentarios es,en "Titulo" [#hex,#hex,...]
# Un archivo:
#   streamlit run tablero.py -- unificado.csv ColumnaFuente EtiquetaDMO es,en "Titulo" [#hex,#hex,...]
#
# (el "--" separa los argumentos de Streamlit de los del tablero)
#
# El análisis lo hace final1.py (a través de motor.py); este archivo solo arma la interfaz.

import argparse
import html
import os
import re
import sys

import streamlit as st

import motor

# ----------------------------------------------------------------- argumentos
EXT = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet")
USO = """**Uso desde la terminal**

Dos archivos:
`streamlit run tablero.py -- dmo.csv comparar.csv ColumnaComentarios es,en "Mi tablero" [#0072B2,#E69F00]`

Un archivo:
`streamlit run tablero.py -- unificado.csv ColumnaFuente EtiquetaDMO es,en "Mi tablero" [#0072B2,#E69F00]`
"""


def es_archivo(x):
    return x.lower().endswith(EXT) or os.path.isfile(x)


def leer_argumentos(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("pos", nargs="*")
    p.add_argument("--col-texto", default=None, help="columna de comentarios (modo un archivo)")
    p.add_argument("--contaminacion", type=float, default=0.05)
    p.add_argument("--max-por-fuente", type=int, default=6000)
    p.add_argument("--paleta", default="cividis", choices=motor.F.PALETAS_CONTINUAS)
    a, _ = p.parse_known_args(argv)
    pos = a.pos
    if len(pos) < 3:
        return None
    cfg = {"contaminacion": a.contaminacion, "max_por_fuente": a.max_por_fuente, "paleta": a.paleta}
    if len(pos) >= 2 and es_archivo(pos[1]):          # dos archivos
        if len(pos) < 4:
            return None
        cfg.update(rutas=pos[:2], col_texto=pos[2], col_fuente=None, etiqueta_dmo=None)
        resto = pos[3:]
    else:                                              # un archivo
        cfg.update(rutas=pos[:1], col_texto=a.col_texto, col_fuente=pos[1], etiqueta_dmo=pos[2])
        resto = pos[3:]
    cfg["idiomas"] = [x for x in re.split(r"[,; ]+", resto[0]) if x] if resto else ["es"]
    cfg["titulo"] = resto[1] if len(resto) > 1 else "Imagen proyectada vs percibida"
    cfg["colores"] = [c for c in re.split(r"[,; ]+", resto[2]) if c] if len(resto) > 2 else []
    return cfg


# ----------------------------------------------------------------- estilo
NAVY, GOLD, INK, MUTED, BG = "#102A43", "#D6A62E", "#17212B", "#627D98", "#F5F7FA"

cfg = leer_argumentos(sys.argv[1:])
TITULO = cfg["titulo"] if cfg else "Imagen proyectada vs percibida"

st.set_page_config(page_title=TITULO, page_icon="🧭", layout="wide")
st.markdown(f"""
<style>
 html, body, [data-testid="stAppViewContainer"] {{ background:{BG}; color:{INK}; }}
 [data-testid="stSidebar"] {{ background:{GOLD}; border-right:3px solid {NAVY}; }}
 [data-testid="stSidebar"] * {{ color:{NAVY} !important; }}
 .main-title {{ background:{NAVY}; color:white; padding:17px 22px; border-radius:10px; text-align:center;
               font-size:25px; font-weight:750; margin-bottom:7px; }}
 .subtitle {{ color:{NAVY}; text-align:center; font-size:14px; margin:0 0 18px 0; }}
 .section-band {{ background:{NAVY}; color:white; border-left:6px solid {GOLD}; border-radius:8px;
                 padding:9px 14px; font-weight:700; font-size:17px; margin:22px 0 10px 0; }}
 .section-band.first {{ margin-top:6px; }}
 .metric-card {{ background:white; border:1px solid #D8E0E8; border-top:4px solid {GOLD}; border-radius:10px;
                padding:12px 14px; box-shadow:0 2px 8px rgba(16,42,67,.06); }}
 .metric-label {{ color:{MUTED}; font-size:13px; font-weight:650; }}
 .metric-value {{ color:{NAVY}; font-size:26px; font-weight:800; }}
 .ejemplo {{ background:white; border-left:5px solid {GOLD}; border-radius:6px; padding:10px 14px;
            margin:8px 0; font-size:14px; line-height:1.45; }}
 .ejemplo small {{ color:{MUTED}; }}
 .ejemplo.pos {{ border-left-color:#009E73; }}
 .ejemplo.neg {{ border-left-color:#D55E00; }}
 .ejemplo.dmo {{ border-left-color:#0072B2; }}
 .ejemplo.perc {{ border-left-color:#E69F00; }}
 .topico-nombre {{ background:{NAVY}; color:white; border-radius:6px; padding:6px 12px; font-weight:700;
                  font-size:15px; margin:14px 0 2px 0; }}
 .topico-nombre span {{ color:{GOLD}; }}
 .nota {{ color:{MUTED}; font-size:13px; margin:2px 0 8px 0; }}
 h1,h2,h3,h4 {{ color:{NAVY}; }}
</style>""", unsafe_allow_html=True)

st.markdown(f'<div class="main-title">{html.escape(TITULO)}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Comparación de la imagen proyectada (DMO) y la imagen percibida</div>',
            unsafe_allow_html=True)
if cfg is None:
    st.warning("Faltan argumentos para iniciar el tablero.")
    st.markdown(USO)
    st.stop()


# ----------------------------------------------------------------- utilidades de interfaz
def banda(titulo, descripcion=None, primera=False):
    st.markdown(f'<div class="section-band{" first" if primera else ""}">{titulo}</div>',
                unsafe_allow_html=True)
    if descripcion:
        st.markdown(f'<div class="nota">{descripcion}</div>', unsafe_allow_html=True)


def metrica(col, etiqueta, valor):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{etiqueta}</div>'
                 f'<div class="metric-value">{valor}</div></div>', unsafe_allow_html=True)


def img(png, caption=None):
    """Imagen PNG (200 dpi) ajustada al ancho de su columna, sin reescalar hacia arriba."""
    if png is None:
        return
    try:
        st.image(png, width="stretch", caption=caption)
    except Exception:                       # versiones de Streamlit anteriores
        st.image(png, use_container_width=True, caption=caption)


def fig(nombre, caption=None):
    png = R["figs"].get(nombre)
    if png is None:
        st.info("Este gráfico no se pudo generar con los datos actuales (faltan comentarios en algún grupo).")
        return
    img(png, caption)


def ejemplos_ui(lista, titulo, clase="", pie=None):
    if not lista:
        st.caption("Sin ejemplos disponibles para este apartado.")
        return
    st.markdown(f"**{titulo}**")
    for e in lista:
        extra = []
        if e.get("polaridad") is not None:
            extra.append(f"polaridad {e['polaridad']:+.2f}")
        if e.get("similitud") is not None:
            extra.append(f"similitud con el concepto {e['similitud']:.2f}")
        if e.get("faceta"):
            extra.append(f"faceta: {html.escape(e['faceta'])}")
        if e.get("caracteres") is not None:
            extra.append(f"{e['caracteres']:,} caracteres (mediana de su fuente: {e['mediana']:,})")
        if e.get("motivo"):
            extra.append(f"término más distintivo: <b>{html.escape(e['motivo'])}</b>")
        if e.get("score") is not None:
            extra.append(f"score de anomalía {e['score']:.2f}")
        txt = html.escape(e["texto"])
        st.markdown(f'<div class="ejemplo {clase}">“{txt}”<br><small>Fuente: {html.escape(e["fuente"])}'
                    + (" · " + " · ".join(extra) if extra else "") + "</small></div>",
                    unsafe_allow_html=True)


# ----------------------------------------------------------------- opciones y análisis (con caché)
with st.sidebar:
    st.markdown("### Opciones")
    contam = st.slider("Proporción esperada de atípicos", 0.01, 0.15, float(cfg["contaminacion"]), 0.01)
    maxf = st.number_input("Máx. comentarios por fuente", 500, 50000, int(cfg["max_por_fuente"]), 500)
    hay_bert = motor.bert_disponible()
    usar_bert = st.checkbox("Sentimiento con BERT + léxico", hay_bert, disabled=not hay_bert)
    if not hay_bert:
        st.caption("torch/transformers no están instalados: el sentimiento usa solo el léxico. "
                   "Instálalos con `pip install torch transformers` para el clasificador híbrido.")
    dmo_pos = st.checkbox("Tratar a la DMO como positiva", True)
    paletas = list(motor.F.PALETAS_CONTINUAS)
    paleta = st.selectbox("Paleta de los mapas de calor", paletas, index=paletas.index(cfg["paleta"]))
    st.caption("Idiomas: " + ", ".join(cfg["idiomas"]))
    st.caption("Paletas: Okabe-Ito (categorías) y la elegida arriba (mapas de calor), con tramas y marcadores redundantes. "
               "Gráficos a 200 dpi.")


@st.cache_resource(show_spinner=False)
def correr(rutas, col_texto, col_fuente, etiqueta, idiomas, colores, contam, maxf, bert, dmo_pos, paleta, mtimes):
    return motor.ejecutar(list(rutas), col_texto=col_texto, col_fuente=col_fuente, etiqueta_dmo=etiqueta,
                          idiomas=list(idiomas), contaminacion=contam, max_por_fuente=maxf,
                          usar_bert=bert, dmo_positiva=dmo_pos, colores=list(colores), paleta=paleta)


try:
    mt = tuple(os.path.getmtime(r) for r in cfg["rutas"])
    with st.spinner("Analizando los comentarios… la primera vez puede tardar unos minutos "
                    "(con BERT, más); después los resultados quedan en caché."):
        R = correr(tuple(cfg["rutas"]), cfg["col_texto"], cfg["col_fuente"], cfg["etiqueta_dmo"],
                   tuple(cfg["idiomas"]), tuple(cfg["colores"]), contam, int(maxf), usar_bert, dmo_pos, paleta, mt)
except SystemExit as e:                      # final1 usa SystemExit para errores de entrada
    st.error(f"No se pudo ejecutar el análisis: {e}")
    st.stop()
except Exception as e:
    st.error(f"No se pudo ejecutar el análisis: {e}")
    st.stop()

c = st.columns(5)
metrica(c[0], "Comentarios analizados", f"{R['n_total']:,}")
metrica(c[1], "Fuentes", len(R["fuentes"]))
metrica(c[2], "Atípicos", f"{R['n_atipicos']:,}")
metrica(c[3], "% positivo (percibida)", "—" if R["pct_pos_perc"] is None else f"{R['pct_pos_perc']:.0f}%")
metrica(c[4], "% negativo (percibida)", "—" if R["pct_neg_perc"] is None else f"{R['pct_neg_perc']:.0f}%")
st.caption(f"DMO: **{R['dmo']}** · clasificador de sentimiento: **{R['metodo']}** · porcentajes sobre "
           f"{R['n_clasif_perc']:,} comentarios percibidos clasificados · "
           "los atípicos se excluyen del análisis de sentimiento y tópicos.")

tabs = st.tabs(["1 · Comportamiento", "2 · Análisis de sentimiento", "3 · Tópicos",
                "4 · Imagen proyectada vs percibida", "5 · Precio, valor y costo"])

# =================================================================== 1. COMPORTAMIENTO
with tabs[0]:
    banda("Comportamiento de los valores atípicos",
          "Cada punto es un comentario proyectado a 2 dimensiones (SVD sobre TF-IDF). "
          "Las ✕ naranja son los atípicos detectados con Isolation Forest dentro de cada fuente.",
          primera=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Todas las fuentes**")
        fig("disp_global")
    with b:
        if R["disp_fuente"]:
            f_sel = st.selectbox("Dispersión de una fuente", list(R["disp_fuente"]), key="disp_f")
            img(R["disp_fuente"][f_sel])
    a, b = st.columns(2)
    with a:
        fig("atipicos_pct")
    with b:
        fig("atipicos_score")
    ejemplos_ui(R["ej_atip"].get("Todas las fuentes"), "Ejemplo de comentarios atípicos (los más anómalos)")
    otras_f = [f for f in R["ej_atip"] if f != "Todas las fuentes"]
    if otras_f:
        f_at = st.selectbox("Ver atípicos de una fuente", otras_f, key="atip_f")
        ejemplos_ui(R["ej_atip"][f_at], f"Atípicos de {f_at}")

    banda("Bigramas y trigramas",
          "Las expresiones de dos y tres palabras que más se repiten en los comentarios normales (sin atípicos).")
    img(R["ngram_global"])
    f_ng = st.selectbox("Bigramas y trigramas de una fuente", list(R["ngram_fuente"]), key="ng_f")
    img(R["ngram_fuente"][f_ng])

# =================================================================== 2. SENTIMIENTO
with tabs[1]:
    banda("Nubes de palabras: comentarios positivos y negativos",
          "Tamaño de la palabra = frecuencia. Verde = positivos, naranja = negativos (comentarios normales).",
          primera=True)
    a, b = st.columns(2)
    for col, s in ((a, "positivo"), (b, "negativo")):
        with col:
            n_s = R["topicos"][s]["n"]
            st.markdown(f"**Comentarios {s}s** (n={n_s:,})")
            if R["nubes"].get(s):
                img(R["nubes"][s])
            elif R["nubes_barras"].get(s):
                img(R["nubes_barras"][s])
                st.caption("Instala `wordcloud` para ver la nube; mientras tanto se muestran barras.")
            else:
                st.caption("Sin suficientes comentarios.")

    banda("Clasificación de sentimiento por fuente",
          f"Comentarios normales clasificados como positivos o negativos con el método: <b>{R['metodo']}</b>. "
          "La DMO se asume positiva por diseño.")
    fig("01_sentimiento_por_fuente")
    ejemplos_ui(R["ej_sent"]["positivo"], "Ejemplos positivos (imagen percibida)", "pos")
    ejemplos_ui(R["ej_sent"]["negativo"], "Ejemplos negativos (imagen percibida)", "neg")

# =================================================================== 3. TÓPICOS
with tabs[2]:
    primera = True
    for s, pref, titulo_s in (("positivo", "pos", "positivos"), ("negativo", "neg", "negativos")):
        T = R["topicos"][s]
        banda(f"Clusters de comentarios {titulo_s} (n={T['n']:,})",
              "Cada color/forma es un tópico (NMF sobre TF-IDF); el número en el círculo marca su centro. "
              "El número de tópicos se elige por coherencia NPMI.", primera=primera)
        primera = False
        if T["enfoque"] == "ninguno":
            st.info(f"No hay comentarios {titulo_s} para analizar.")
            continue
        if T["enfoque"] == "frecuencias":
            st.info(f"Hay pocos comentarios {titulo_s}: no alcanzan para estimar tópicos estables, "
                    "así que se muestran las palabras más frecuentes.")
            fig(f"{pref}_1_frecuencias")
            fig(f"{pref}_2_nube")
            continue
        a, b = st.columns(2)
        with a:
            fig(f"{pref}_4_mapa")
        with b:
            fig(f"{pref}_1_coherencia")
            st.caption("La estrella marca el k elegido: el de mayor coherencia NPMI entre los probados.")
        fig(f"{pref}_2_terminos")
        st.markdown(f"#### Nombre y ejemplo de cada cluster ({titulo_s})")
        st.caption("El nombre del cluster son sus tres términos más característicos.")
        items = list(T["topicos"].items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for col, (t, d) in zip(cols, items[i:i + 2]):
                with col:
                    st.markdown(f'<div class="topico-nombre">T{t} · {html.escape(d["nombre"])} '
                                f'<span>(n={d["n"]:,})</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="nota">Términos: {html.escape(d["terminos"])}</div>',
                                unsafe_allow_html=True)
                    ejemplos_ui(d["ejemplos"][:1], "Ejemplo", "pos" if s == "positivo" else "neg")

# =================================================================== 4. PROYECTADA VS PERCIBIDA
with tabs[3]:
    banda("Tópicos compartidos: imagen proyectada (DMO) vs imagen percibida",
          "Un mismo modelo de tópicos para todos los comentarios; se compara qué % de cada grupo cae en cada tópico. "
          "Trama diagonal = DMO, puntos = imagen percibida.", primera=True)
    fig("06_comparacion_topicos_DMO_percibida")
    if R["ej_comp"]:
        st.markdown("**Ejemplos de cada tópico compartido**")
        for e in R["ej_comp"]:
            with st.expander(f"Tópico {e['topico']}: {e['terminos']} — {e['interpretacion']} "
                             f"(DMO {e['dmo_pct']:.1f}% · percibida {e['perc_pct']:.1f}%)",
                             expanded=e["topico"] == 1):
                x, y = st.columns(2)
                with x:
                    ejemplos_ui(e["dmo"], "Imagen proyectada (DMO)", "dmo")
                with y:
                    ejemplos_ui(e["perc"], "Imagen percibida", "perc")

    banda("Tópicos y fuentes: comentarios positivos y negativos",
          "A la izquierda, cuántos comentarios de cada fuente hay en cada tópico; a la derecha, el % de cada fuente "
          "que cae en el tópico. Los tópicos positivos y negativos vienen de modelos distintos "
          "(T1 positivo no es T1 negativo).")
    st.markdown("**Positivos**")
    if R["topicos"]["positivo"]["enfoque"] == "tópicos":
        fig("pos_3_topicos_fuentes")
    else:
        st.info("La partición positiva no tiene modelo de tópicos.")
    st.markdown("**Negativos**")
    if R["topicos"]["negativo"]["enfoque"] == "tópicos":
        fig("neg_3_topicos_fuentes")
    else:
        st.info("La partición negativa no tiene modelo de tópicos (muy pocos comentarios negativos).")
    if "12_comparacion_topicos_buenos_malos" in R["figs"]:
        fig("12_comparacion_topicos_buenos_malos")

    banda("Comparación semántica global",
          "Matrices de similitud coseno entre los centroides de cada fuente (1 = idénticas) en dos espacios "
          "vectoriales: TF-IDF y LSA.")
    fig("09_similitud_matriz_fuentes")
    a, b = st.columns([3, 2])
    with a:
        fig("11_indice_alineacion")
    with b:
        st.markdown("**Cómo leerlo**")
        st.markdown(
            "- **DMO vs percibida:** qué tanto se parecen en promedio los mensajes de la DMO y los de los usuarios.\n"
            "- **DMO vs DMO (línea base):** qué tanto se parece la DMO a sí misma, con muestras del mismo tamaño.\n"
            "- **Índice de alineación** = el primero entre el segundo. 1 significa que los usuarios hablan "
            "como la DMO; ≥ 0.8 alta, 0.5–0.8 media y < 0.5 baja (cortes orientativos).")
        gl = getattr(R["A"], "similitud_global", None)
        if gl is not None and len(gl):
            for r in gl.itertuples():
                st.markdown(f"- {r.espacio}: índice **{r.indice_alineacion_media:.2f}** ({r.alineacion})")

# =================================================================== 5. PRECIO, VALOR Y COSTO
with tabs[4]:
    banda("Frecuencia y tono del concepto «precio / valor / costo»",
          "Un comentario se considera relacionado cuando su similitud con el concepto supera un umbral "
          "(media + 1.5 desv. est.). Cuidado con los grupos con pocos comentarios relacionados: "
          "revisa la n de cada barra.", primera=True)
    if "precio_frecuencia" not in R["figs"]:
        st.info("No se pudo analizar el concepto: se necesitan al menos 2 comentarios en la DMO y 2 en la percibida.")
    else:
        a, b = st.columns(2)
        with a:
            fig("precio_frecuencia")
        with b:
            fig("precio_tono")
        banda("Similitud con precio / valor / costo vs tono de cada comentario",
              "Cada punto es un comentario: a la derecha de la línea punteada están los relacionados con el "
              "concepto; arriba, los más positivos. Forma y color = fuente.")
        fig("13b_precio_similitud_vs_tono")
        a, b = st.columns(2)
        with a:
            ejemplos_ui(R["ej_precio"]["dmo"], "Comentarios de la DMO más cercanos al concepto", "dmo")
        with b:
            ejemplos_ui(R["ej_precio"]["perc"], "Comentarios percibidos más cercanos al concepto", "perc")

with st.expander("Registro del análisis (consola de final1.py)"):
    st.code(R["log"][-8000:] or "(vacío)")