import time
import streamlit as st
from src import config
from src.ingestion import ingest

st.set_page_config(page_title="Self-Correcting RAG", page_icon="🔁", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 900px;}
.hero h1 {font-size: 2rem; margin-bottom: 0;}
.hero p {color: #8B93A7; margin-top: .2rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem;
       background:#1D2433; color:#AAB4CF; margin-right:6px;}
.pill.ok {background:#143225; color:#6EE7A8;} .pill.warn {background:#3A2A12; color:#F5C26B;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (one-time, ~20s)...")
def load_backend():
    from src.graph import build_graph
    from src.ingestion import get_vectorstore
    get_vectorstore().similarity_search("warmup", k=1)   # pre-load embedder so 1st question is fast
    return build_graph()


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    verify = st.toggle("Verify answer (hallucination check)", value=True,
                       help="Turn off for faster answers.")
    max_retries = st.slider("Max self-correction retries", 0, 3, config.DEFAULT_MAX_RETRIES)

    st.header("📄 Documents")
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    for p in pdfs:
        st.caption(f"• {p.name}")
    if not pdfs:
        st.caption("No PDFs yet.")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if st.button("📥 Index documents", use_container_width=True):
        for f in uploads or []:
            (config.DATA_DIR / f.name).write_bytes(f.getbuffer())
        try:
            with st.spinner("Chunking & embedding..."):
                st.cache_resource.clear()
                st.success(f"Indexed {ingest()} chunks")
        except FileNotFoundError as e:
            st.error(str(e))

    st.header("🔑 Keys")
    for name, val in [("Groq", config.GROQ_API_KEY), ("Tavily", config.TAVILY_API_KEY)]:
        st.markdown(f"<span class='pill {'ok' if val else 'warn'}'>{name}: "
                    f"{'loaded' if val else 'missing'}</span>", unsafe_allow_html=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------- header ----------------
st.markdown("<div class='hero'><h1>🔁 Self-Correcting RAG</h1>"
            "<p>Ask your documents. The system grades its own retrieval and checks its own answers.</p></div>",
            unsafe_allow_html=True)

try:
    graph = load_backend()
except Exception as e:
    st.error(f"Couldn't start the backend: {e}\n\nCheck your `.env` keys and restart.")
    st.stop()

from src.graph import initial_state  # noqa: E402

st.session_state.setdefault("history", [])


def render_extras(m):
    if m.get("trace"):
        with st.expander(f"🧭 How I answered · {m['time']:.1f}s"):
            for line in m["trace"]:
                st.markdown(line)
    if m.get("sources"):
        with st.expander(f"📚 Sources ({len(m['sources'])})"):
            for s in m["sources"]:
                st.markdown(f"**{s['label']}**")
                st.caption(s["text"])


# ---------------- history ----------------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if not m.get("grounded", True):
                st.warning("Could not fully verify this answer against the sources.")
            render_extras(m)

# ---------------- example prompts ----------------
if not st.session_state.history:
    st.markdown("**Try one:**")
    cols = st.columns(3)
    examples = ["Summarize this document", "What are the key steps described?", "Who won the 2022 World Cup?"]
    for c, q in zip(cols, examples):
        if c.button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

question = st.chat_input("Ask something about your documents...") or st.session_state.pop("pending", None)

# ---------------- run ----------------
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=True)
        slot = st.empty()
        state = initial_state(question, max_retries, verify)
        trace, buf, last_step = [], "", None
        t0 = last = time.time()
        try:
            for mode, data in graph.stream(state, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    chunk, meta = data
                    if meta.get("langgraph_node") == "generate" and isinstance(chunk.content, str) and chunk.content:
                        if meta.get("langgraph_step") != last_step:
                            buf, last_step = "", meta.get("langgraph_step")
                        buf += chunk.content
                        slot.markdown(buf + " ▌")
                else:
                    now = time.time()
                    for _, delta in data.items():
                        delta = delta or {}
                        for line in delta.get("trace", []):
                            status.write(f"{line} · `{now - last:.1f}s`")
                            trace.append(f"{line} · `{now - last:.1f}s`")
                        state.update({k: v for k, v in delta.items() if k != "trace"})
                    last = now
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

        total = time.time() - t0
        status.update(label=f"Done in {total:.1f}s", state="complete", expanded=False)
        slot.markdown(state["generation"])
        if not state["grounded"]:
            st.warning("Could not fully verify this answer against the sources.")

        sources = [{"label": str(d.metadata.get("source", "web")).replace("\\", "/").split("/")[-1]
                    + (f" — p.{d.metadata['page'] + 1}" if isinstance(d.metadata.get("page"), int) else ""),
                    "text": d.page_content[:300] + "…"} for d in state["documents"]]
        msg = {"role": "assistant", "content": state["generation"], "trace": trace,
               "sources": sources, "time": total, "grounded": state["grounded"]}
        render_extras(msg)
    st.session_state.history.append(msg)