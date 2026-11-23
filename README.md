# Dashboard-NLP
# Imagen proyectada (DMO) vs imagen percibida — Dashboard de PLN

Compara lo que comunica una DMO con lo que dicen los usuarios en otras fuentes digitales:
limpieza, atípicos, sentimiento, tópicos por sentimiento, brechas, alineación semántica
y el concepto precio/valor/costo. Todo con TF-IDF + LSA (sin APIs ni modelos externos).
Funciona con cualquier archivo de texto/tabla (csv, tsv, txt, json/jsonl, xlsx, parquet).

## Instalación (Mac / Windows)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
```bash
streamlit run app.py
```

1. Sube tus archivos en la barra lateral (un archivo con columna de fuente, o un archivo por fuente).
2. Elige la columna de comentarios, la columna de fuente y cuál(es) fuente(s) son la DMO.
3. Pulsa **Ejecutar análisis**. Los mapas t-SNE se calculan aparte (pestaña "Mapas").

Para probar: `datos_ejemplo.csv` (columnas `comentario`, `fuente`; la DMO es `dmo`).

## Estructura

- `app.py`: interfaz Streamlit.
- `analisis.py`: lógica de análisis (sin interfaz; reutilizable desde un script de línea de comandos).
- `requirements.txt`, `datos_ejemplo.csv`.
