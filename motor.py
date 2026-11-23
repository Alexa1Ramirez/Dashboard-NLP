# -*- coding: utf-8 -*-
"""
motor.py — ejecuta el pipeline de final1.py y deja todo listo para el tablero.

Qué hace:
  1. Corre los mismos pasos que main() de final1.py (limpieza, atípicos, sentimiento,
     tópicos, DMO vs percibida, precio/valor/costo) sin imprimir nada en pantalla.
  2. Captura las figuras de matplotlib y las convierte a PNG de alta resolución
     (200 dpi), de modo que en el tablero se vean nítidas y no pixeleadas.
  3. Prepara los ejemplos de texto de cada apartado.

No necesita Streamlit: puede probarse desde un script (ver `ejecutar`).
"""
import contextlib
import io
import os
import re
import sys
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

import final1 as F

DPI = 200
FONT_MIN = 10          # ningún texto de las figuras baja de este tamaño (pt)
RC = {
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "legend.title_fontsize": 11, "figure.titlesize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8A99A8", "axes.labelcolor": "#17212B",
    "text.color": "#17212B", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
}
_COLORES_BASE = (F.C_DMO, F.C_PERC, list(F.C_FUENTES))
_LOCK = threading.Lock()          # final1 usa estado global (FIGS, RESUMEN, pyplot)
C_BI, H_BI = "#56B4E9", ".."      # bigramas
C_TRI, H_TRI = "#CC79A7", "xx"    # trigramas


# ------------------------------------------------------------------ utilidades gráficas
def _png(fig):
    """Figura -> bytes PNG a 200 dpi, con tipografía mínima legible."""
    for t in fig.findobj(Text):
        if t.get_text() and t.get_fontsize() < FONT_MIN:
            t.set_fontsize(FONT_MIN)
    try:
        fig.tight_layout()
    except Exception:
        pass
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _nombres_topicos(part):
    """{n_topico: 'palabra1, palabra2, palabra3'} a partir de los términos del NMF."""
    terminos = part.attrs.get("terminos", []) if part is not None else []
    return {i + 1: ", ".join(w for w, _ in t[:3]) for i, t in enumerate(terminos)}


_RE_TOP = re.compile(r"^Tópico\s+(\d+)\s*(.*)$", re.S)


def _etiquetar_topicos(fig, nombres):
    """Reemplaza 'Tópico 3' por 'T3 · museo, pena, visitar' en títulos, ejes y leyendas."""
    if not nombres:
        return

    def nuevo(txt):
        m = _RE_TOP.match(str(txt).strip())
        if not m or int(m.group(1)) not in nombres:
            return None
        n, resto = int(m.group(1)), m.group(2).strip()
        return f"T{n} · {nombres[n]}" + (f" {resto}" if resto else "")

    fig.canvas.draw()
    for ax in fig.axes:
        t = nuevo(ax.get_title())
        if t:
            ax.set_title(t, fontsize=12, fontweight="bold")
        lg = ax.get_legend()
        if lg is not None:
            for tx in lg.get_texts():
                t = nuevo(tx.get_text())
                if t:
                    tx.set_text(t)
        ylab = [l.get_text() for l in ax.get_yticklabels()]
        if any(nuevo(l) for l in ylab):
            ax.set_yticklabels([nuevo(l) or l for l in ylab])


def _fig_ngramas(textos, titulo):
    """Bigramas y trigramas más frecuentes (barras con color + trama)."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, n, nom, color, hatch in zip(axs, (2, 3), ("Bigramas", "Trigramas"),
                                        (C_BI, C_TRI), (H_BI, H_TRI)):
        t = F.tabla_ngramas(textos, n, 15)
        if t.empty:
            ax.set_title(f"{nom} (sin datos)")
            ax.axis("off")
            continue
        t = t.sort_values("frecuencia")
        barras = ax.barh(t["termino"], t["frecuencia"], color=color, edgecolor="black", hatch=hatch)
        for b, v in zip(barras, t["frecuencia"]):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {int(v)}", va="center", fontsize=10)
        ax.set_title(nom, fontweight="bold")
        ax.set_xlabel("Frecuencia")
        ax.margins(x=.12)
    fig.suptitle(titulo, fontweight="bold")
    return fig


def _nube_png(textos, color, semilla):
    """Nube de palabras en alta resolución (2800 x 1600 px) o None si falta wordcloud."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    texto = " ".join(pd.Series(textos).dropna().astype(str))
    if not texto.strip():
        return None
    nube = WordCloud(width=1400, height=800, scale=2, background_color="white", max_words=100,
                     collocations=False, random_state=semilla, prefer_horizontal=.9,
                     color_func=lambda *a, _c=color, **k: _c).generate(texto)
    buf = io.BytesIO()
    nube.to_image().save(buf, format="PNG")
    return buf.getvalue()


def _fig_barras_palabras(textos, color, hatch, titulo):
    """Alternativa a la nube cuando wordcloud no está instalado."""
    t = F.tabla_ngramas(textos, 1, 20)
    fig, ax = plt.subplots(figsize=(8, 7))
    if t.empty:
        ax.axis("off")
        return fig
    t = t.sort_values("frecuencia")
    ax.barh(t["termino"], t["frecuencia"], color=color, edgecolor="black", hatch=hatch)
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Frecuencia")
    return fig


def _fig_precio_frecuencia(frec, A):
    fig, ax = plt.subplots(figsize=(9, 6))
    etiquetas, i_otra = [], 0
    for j, r in frec.reset_index(drop=True).iterrows():
        if r["nombre"] == F.G_DMO or (r["nivel"] == "fuente" and r["nombre"] == str(A.dmo)):
            col, h = F.C_DMO, F.H_DMO
        elif r["nombre"] == F.G_PERC:
            col, h = F.C_PERC, F.H_PERC
        else:
            col, h = F.C_FUENTES[i_otra % len(F.C_FUENTES)], F.HATCHES[i_otra % len(F.HATCHES)]
            i_otra += 1
        ax.bar(j, r["pct_relacionados"], .7, color=col, edgecolor="black", hatch=h,
               yerr=[[r["pct_relacionados"] - r["IC95_inf"]], [r["IC95_sup"] - r["pct_relacionados"]]],
               capsize=5)
        ax.text(j, r["IC95_sup"] * 1.02 + .1, f"{r['pct_relacionados']:.1f}%", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
        etiquetas.append("\n".join(_envolver(str(r["nombre"]), 12)))
    ax.set_xticks(range(len(frec)))
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("% de comentarios relacionados (IC 95 % de Wilson)")
    ax.set_title("Frecuencia del concepto «precio / valor / costo»", fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.margins(y=.15)
    return fig


def _fig_precio_tono(tono):
    fig, ax = plt.subplots(figsize=(9, 6))
    w = .38
    for gi, (gnom, col, h) in enumerate(((F.G_DMO, F.C_DMO, F.H_DMO), (F.G_PERC, F.C_PERC, F.H_PERC))):
        for k, (sub_nom, alpha, hh) in enumerate((("relacionados con precio", .95, h),
                                                  ("resto de comentarios", .35, ""))):
            f = tono[(tono["grupo"] == gnom) & (tono["subconjunto"] == sub_nom)]
            if f.empty:
                continue
            f = f.iloc[0]
            v, xpos = f["media"], gi + (k - .5) * w
            if np.isfinite(v):
                ax.bar(xpos, v, w, color=col, alpha=alpha, edgecolor="black", hatch=hh)
                ax.text(xpos, v + (.012 if v >= 0 else -.012), f"{v:+.2f}\n(n={int(f['n'])})",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
    ax.axhline(0, color="black", lw=.8)
    ax.margins(y=.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([F.G_DMO, F.G_PERC])
    ax.set_ylabel("Polaridad media (−1 negativo … +1 positivo)")
    ax.set_title("Tono: comentarios de precio (trama) vs resto (claro)", fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    return fig


def _envolver(s, n):
    import textwrap
    return textwrap.wrap(s, n) or [s]


# ------------------------------------------------------------------ ejemplos de texto
def _elegir(df, n=3, orden=None, asc=False, terminos=None, lo=70, hi=420):
    """Elige n comentarios legibles (longitud moderada), variando la fuente si se puede."""
    if df is None or len(df) == 0:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    d = df
    ok = d[d["texto_original"].astype(str).str.len().between(lo, hi)]
    if len(ok) >= n:
        d = ok
    if terminos:
        tl = d["texto_limpio"].fillna("").astype(str)
        d = d.assign(_s=tl.map(lambda t: sum(w in t for w in terminos))).sort_values("_s", ascending=False)
    elif orden:
        d = d.sort_values(orden, ascending=asc)
    primeras = d.drop_duplicates("fuente").head(n)
    if len(primeras) < n:
        primeras = pd.concat([primeras, d.drop(primeras.index).head(n - len(primeras))])
    return primeras


def _ej(r, largo=380, **extra):
    return {"texto": F.recortar(r.texto_original, largo), "fuente": str(r.fuente), **extra}


def _pol(v):
    return None if v is None or not np.isfinite(v) else float(v)


# ------------------------------------------------------------------ ejecución
def _validar_colores(colores):
    return [c if c.startswith("#") else "#" + c for c in (colores or []) if re.fullmatch(r"#?[0-9A-Fa-f]{6}", c)]


def _aplicar_colores(colores):
    F.C_DMO, F.C_PERC, F.C_FUENTES = _COLORES_BASE[0], _COLORES_BASE[1], list(_COLORES_BASE[2])
    cols = _validar_colores(colores)
    if cols:
        F.C_DMO = cols[0]
    if len(cols) > 1:
        F.C_PERC = cols[1]
    if len(cols) > 2:
        F.C_FUENTES = cols[2:] + F.C_FUENTES


def _parsear(argv):
    viejo = sys.argv
    sys.argv = argv
    try:
        return F.argumentos()
    finally:
        sys.argv = viejo


def bert_disponible():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def ejecutar(rutas, col_texto=None, col_fuente=None, etiqueta_dmo=None, idiomas=("es",),
             contaminacion=0.05, max_por_fuente=6000, usar_bert=True, dmo_positiva=True,
             colores=(), n_boot=100, paleta="cividis"):
    """Corre todo el análisis y devuelve un diccionario con PNG, tablas y ejemplos."""
    rutas = list(rutas)
    if etiqueta_dmo is None and len(rutas) >= 2:        # modo dos archivos: la DMO es el primero
        etiqueta_dmo = os.path.splitext(os.path.basename(rutas[0]))[0]
    argv = ["final1", *rutas, "--max-por-fuente", str(int(max_por_fuente)),
            "--contaminacion", str(float(contaminacion)), "--n-boot", str(int(n_boot)),
            "--idioma", idiomas[0] if len(idiomas) == 1 else "auto", "--paleta", paleta]
    if col_texto:
        argv += ["--col-texto", col_texto]
    if col_fuente:
        argv += ["--col-fuente", col_fuente]
    if etiqueta_dmo:
        argv += ["--dmo", etiqueta_dmo]
    if not usar_bert:
        argv += ["--sin-bert"]
    if not dmo_positiva:
        argv += ["--dmo-puede-ser-negativa"]

    buf = io.StringIO()
    with _LOCK, plt.rc_context(RC), contextlib.redirect_stdout(buf):
        plt.close("all")
        F.LOG.clear()
        F.RESUMEN.clear()
        F.FIGS.clear()
        _aplicar_colores(colores)
        A = _parsear(argv)

        # 1-2. carga, limpieza y atípicos
        df = F.ejecutar_base(A)
        antes = set(plt.get_fignums())
        df = F.detectar_isolation_forest_vectorial(df, A)
        figs_atip, disp_fuente = {}, {}
        for n in [n for n in plt.get_fignums() if n not in antes]:
            fig = plt.figure(n)
            if not fig.axes or fig.axes[0].name == "3d":
                continue
            t = fig.axes[0].get_title()
            if t.startswith("Porcentaje de comentarios atípicos"):
                figs_atip["atipicos_pct"] = _png(fig)
            elif t.startswith("Distribución del score"):
                figs_atip["atipicos_score"] = _png(fig)
            elif "TODAS LAS BASES" in t and "2D" in t:
                figs_atip["disp_global"] = _png(fig)
            elif t.startswith("Fuente:") and "2D" in t:
                disp_fuente[t.split("Fuente:")[1].split("—")[0].strip()] = _png(fig)
        plt.close("all")

        n_total, n_out = len(df), int(df["outlier"].sum())
        normales = df.loc[~df["outlier"]].reset_index(drop=True)

        # 3. sentimiento híbrido
        normales = F.clasificar_hibrido(normales, A)
        pos, neg = F.separar_particiones(normales)
        metodo = "BERT + léxico" if normales["pol_bert"].notna().any() else "solo léxico"

        F.FIGS.clear()
        F.g_distribucion_fuentes(normales, A, metodo)

        # 4. DMO vs percibida
        F.comparar_topicos_dmo_percibida(normales, A)
        F.comparar_similitud(normales, A)
        F.analizar_precio_valor_costo(normales, A)

        # 5. tópicos por sentimiento
        pos = F.procesar_particion(pos, "positiva", F.C_POS, F.H_POS, "pos", A, False)
        neg = F.procesar_particion(neg, "negativa", F.C_NEG, F.H_NEG, "neg", A, True)
        F.comparar_topicos_buenos_malos(pos, neg, A)

        # ---- figuras de final1 -> PNG (con nombres de tópico en las de tópicos)
        nom_t = {"pos": _nombres_topicos(pos), "neg": _nombres_topicos(neg)}
        figs = dict(figs_atip)
        for nombre, fig in F.FIGS:
            pref = nombre.split("_")[0]
            if pref in nom_t and nombre.endswith(("_2_terminos", "_3_topicos_fuentes", "_4_mapa")):
                _etiquetar_topicos(fig, nom_t[pref])
            if nombre.startswith("13"):          # en el tablero la leyenda no es interactiva
                for ax in fig.axes:
                    lg = ax.get_legend()
                    if lg is not None:
                        lg.set_title(lg.get_title().get_text().replace(" (clic = ocultar/mostrar)", ""))
            figs[nombre] = _png(fig)
        plt.close("all")
        F.FIGS.clear()

        # ---- figuras propias (ngramas, nubes, precio)
        ngram_global = _png(_fig_ngramas(normales["texto_limpio"],
                                         f"Bigramas y trigramas — todos los comentarios normales (n={len(normales):,})"))
        ngram_fuente = {}
        for f, g in normales.groupby("fuente"):
            ngram_fuente[str(f)] = _png(_fig_ngramas(g["texto_limpio"],
                                                     f"Bigramas y trigramas — {f} (n={len(g):,})"))
        plt.close("all")

        nubes, nubes_barras = {}, {}
        for s, g, c, h in (("positivo", pos, F.C_POS, F.H_POS), ("negativo", neg, F.C_NEG, F.H_NEG)):
            nubes[s] = _nube_png(g["texto_limpio"], c, A.semilla)
            if nubes[s] is None and len(g):
                nubes_barras[s] = _png(_fig_barras_palabras(g["texto_limpio"], c, h,
                                                            f"Palabras más frecuentes — {s}s"))
        plt.close("all")

        if hasattr(A, "precio_frecuencia"):
            figs["precio_frecuencia"] = _png(_fig_precio_frecuencia(A.precio_frecuencia, A))
            figs["precio_tono"] = _png(_fig_precio_tono(A.precio_tono))
            plt.close("all")

    # ------------------------------------------------------------------ tablas y ejemplos
    dmo = str(A.dmo)
    fuentes = [str(f) for f in df["fuente"].drop_duplicates()]
    perc = normales[~normales["es_dmo"]]
    clasif = perc[perc["etiqueta"].isin([F.POS, F.NEG])]

    # atípicos
    ej_atip = {}
    med = df.assign(largo=df["texto_original"].str.len()).groupby("fuente")["largo"].median()
    out = df[df["outlier"]].assign(largo=lambda d: d["texto_original"].str.len())
    for nombre, sub in [("Todas las fuentes", out)] + [(str(f), out[out["fuente"] == f]) for f in fuentes]:
        sel = sub.sort_values("if_score", ascending=False).head(3)
        ej_atip[nombre] = [_ej(r, 520, caracteres=int(r.largo), mediana=int(med.get(r.fuente, 0)),
                               motivo=str(r.if_motivo), score=float(r.if_score)) for r in sel.itertuples()]

    # sentimiento
    pos_p, neg_p = clasif[clasif["etiqueta"] == F.POS], clasif[clasif["etiqueta"] == F.NEG]
    pos_p = pos_p.assign(abs_pol=pos_p["polaridad"].abs())
    neg_p = neg_p.assign(abs_pol=neg_p["polaridad"].abs())
    ej_sent = {
        "positivo": [_ej(r, polaridad=_pol(r.polaridad)) for r in _elegir(pos_p, 3, "abs_pol").itertuples()],
        "negativo": [_ej(r, polaridad=_pol(r.polaridad)) for r in _elegir(neg_p, 3, "abs_pol").itertuples()],
    }

    # tópicos por sentimiento
    topicos = {}
    for s, g, pref in (("positivo", pos, "pos"), ("negativo", neg, "neg")):
        nombres = _nombres_topicos(g)
        enfoque = "ninguno" if len(g) == 0 else ("tópicos" if nombres else "frecuencias")
        ejemplos = {}
        for t, terms in enumerate(g.attrs.get("terminos", []), 1):
            sub = g[g["topico"] == t]
            claves = [w for w, _ in terms[:10]]
            sel = _elegir(sub, 2, terminos=claves) if len(sub) else sub
            ejemplos[t] = {"n": int(len(sub)), "nombre": nombres[t],
                           "terminos": ", ".join(w for w, _ in terms[:8]),
                           "ejemplos": [_ej(r, polaridad=_pol(r.polaridad)) for r in sel.itertuples()]}
        topicos[s] = {"enfoque": enfoque, "n": int(len(g)), "k": len(ejemplos), "topicos": ejemplos}

    # comparación de tópicos compartidos
    ej_comp = []
    if hasattr(A, "comparacion_topicos"):
        datos = A.comentarios_topicos_comparacion
        for i, fila in enumerate(A.comparacion_topicos.itertuples(), 1):
            claves = [w.strip() for w in str(fila.terminos_representativos).split(",")]
            sub = datos[datos["topico_comparacion"] == i]
            d_ = _elegir(sub[sub["grupo_comparacion"] == "DMO"], 1, terminos=claves)
            p_ = _elegir(sub[sub["grupo_comparacion"] != "DMO"], 1, terminos=claves)
            ej_comp.append({"topico": i, "terminos": fila.terminos_representativos,
                            "dmo_pct": float(fila.DMO_pct), "perc_pct": float(fila.percibida_pct),
                            "interpretacion": fila.interpretacion,
                            "dmo": [_ej(r) for r in d_.itertuples()],
                            "perc": [_ej(r) for r in p_.itertuples()]})

    # precio / valor / costo
    ej_precio = {"dmo": [], "perc": []}
    if hasattr(A, "precio_top"):
        tp = A.precio_top
        for clave, grupo in (("dmo", F.G_DMO), ("perc", F.G_PERC)):
            sub = tp[tp["ranking_en"] == grupo].head(3)
            ej_precio[clave] = [{"texto": F.recortar(r.texto_original, 380), "fuente": str(r.fuente),
                                 "similitud": float(r.sim_precio), "polaridad": _pol(r.polaridad),
                                 "faceta": str(r.faceta or "")} for r in sub.itertuples()]

    return {
        "A": A, "dmo": dmo, "fuentes": fuentes, "metodo": metodo,
        "n_total": n_total, "n_atipicos": n_out, "n_normales": len(normales),
        "pct_pos_perc": (100 * (clasif["etiqueta"] == F.POS).mean()) if len(clasif) else None,
        "pct_neg_perc": (100 * (clasif["etiqueta"] == F.NEG).mean()) if len(clasif) else None,
        "n_clasif_perc": int(len(clasif)),
        "figs": figs, "disp_fuente": disp_fuente,
        "ngram_global": ngram_global, "ngram_fuente": ngram_fuente,
        "nubes": nubes, "nubes_barras": nubes_barras,
        "ej_atip": ej_atip, "ej_sent": ej_sent, "topicos": topicos,
        "ej_comp": ej_comp, "ej_precio": ej_precio,
        "n_orig": df.groupby("fuente").size().to_dict(),
        "log": buf.getvalue(),
    }