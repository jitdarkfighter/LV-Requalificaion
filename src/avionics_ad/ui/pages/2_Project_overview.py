"""Project and backend-pipeline overview, rendered from docs/04-platform-overview.md.

The markdown file is the single source; this page only displays it. Image lines
(``![alt](path)``) are shown with ``st.image``, because ``st.markdown`` cannot
resolve a path relative to the document.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

DOC = Path(__file__).resolve().parents[4] / "docs" / "04-platform-overview.md"
IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")

st.set_page_config(page_title="Project overview", page_icon=":material/hub:", layout="wide")

if not DOC.exists():
    st.error(f"Overview document not found: {DOC}")
    st.stop()

chunk: list[str] = []
for line in DOC.read_text().splitlines():
    match = IMAGE.match(line)
    if not match:
        chunk.append(line)
        continue
    st.markdown("\n".join(chunk))
    chunk = []
    image = (DOC.parent / match["path"]).resolve()
    if image.exists():
        st.image(str(image), caption=match["alt"], width="stretch")
    else:
        st.warning(f"Figure missing: {image}. Run `.venv/bin/python docs/figures/make_platform_figure.py`.")
st.markdown("\n".join(chunk))
