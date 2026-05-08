# Urdu font

Place the self-hosted WOFF2 font here:

`NotoNastaliqUrdu-Regular.woff2`

Recommended source: Google Webfonts Helper, WOFF2 only, regular weight.

The Streamlit app serves this directory through `.streamlit/config.toml`, and
`app/main.py` loads the font only for Urdu text via `.urdu` / inline Urdu
blocks. English text stays on system fonts for fastest rendering.
