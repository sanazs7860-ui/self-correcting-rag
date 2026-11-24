import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from . import config
from .ingestion import get_vectorstore
from .state import GraphState, GradeRelevance, AnswerCheck

# ---------------- LLMs ----------------
# Small, fast model: relevance grading + query rewriting
_fast_llm = ChatGroq(
    model=config.GRADER_MODEL,
    temperature=0,
    api_key=config.GROQ_API_KEY,
    reasoning_effort="low",
)


def _make_generator():
    if config.GENERATOR_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.GEMINI_GENERATOR_MODEL, temperature=0)
    return ChatGroq(
        model=config.GROQ_GENERATOR_MODEL,
        temperature=0,
        api_key=config.GROQ_API_KEY,
        reasoning_effort="low",
    )


# Larger model: writes the final answer
_gen_llm = _make_generator()

# Larger model with more thinking: judges hallucinations
_judge_llm = ChatGroq(
    model=config.GROQ_GENERATOR_MODEL,
    temperature=0,
    api_key=config.GROQ_API_KEY,
    reasoning_effort="medium",
)


# ---------------- JSON parsing (replaces tool calling) ----------------
def _json_parser(schema):
    """Finds the first {...} in the model's reply and validates it with Pydantic."""
    def _parse(msg):
        text = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(text, list):
            text = "".join(str(t) for t in text)
        match = re.search(r"\{.*?\}", text or "", re.S)
        if not match:
            raise ValueError(f"No JSON found in reply: {text!r}")
        data = json.loads(match.group(0))
        data = {k: str(v).strip().lower() for k, v in data.items()}
        return schema(**data)
    return RunnableLambda(_parse)


# ---------------- Chains ----------------
relevance_chain = ChatPromptTemplate.from_messages([
    ("system", "Decide if a document chunk contains ANY information related to the user's question, "
               "even partially or indirectly. Answer 'yes' if it is related or might help. "
               "Answer 'no' ONLY if it is clearly about something else. "
               "Reply with ONLY a JSON object and nothing else, for example: "
               "{{\"relevant\": \"yes\"}} or {{\"relevant\": \"no\"}}"),
    ("human", "Question: {question}\n\nChunk:\n{document}"),
]) | _fast_llm | _json_parser(GradeRelevance)

rewrite_chain = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question as a sharper vector-search query for technical manuals "
               "(add likely keywords, remove vagueness). Return ONLY the query."),
    ("human", "Original question: {original}\nLast attempt: {question}"),
]) | _fast_llm

generate_chain = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context. Cite sources inline like [source: file, p.N]. "
               "Be concise. If the context answers only part of the question, give what it does say "
               "and state clearly what is missing. Say you don't know only if the context has "
               "nothing relevant.{strict}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | _gen_llm

check_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fair fact-checker. "
     "grounded = 'yes' if the answer's factual claims are supported by, or reasonably paraphrased "
     "from, the context. Ignore citation tags, formatting and wording. Answer 'no' ONLY if the "
     "answer states a specific fact, number or name that does not appear in the context. "
     "An answer that honestly says the information is not available is grounded. "
     "useful = 'yes' if the answer addresses the question or honestly says it cannot be answered. "
     "Reply with ONLY a JSON object and nothing else, for example: "
     "{{\"grounded\": \"yes\", \"useful\": \"yes\"}}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{generation}"),
]) | _judge_llm | _json_parser(AnswerCheck)


# ---------------- Helpers ----------------
def _label(doc: Document) -> str:
    name = str(doc.metadata.get("source", "web")).split("/")[-1].split("\\")[-1]
    page = doc.metadata.get("page")
    return f"{name}, p.{page + 1}" if isinstance(page, int) else name


def _format_docs(docs):
    return "\n\n".join(f"[source: {_label(d)}]\n{d.page_content}" for d in docs)


# ---------------- Nodes ----------------
def retrieve(state: GraphState):
    docs = get_vectorstore().similarity_search(state["question"], k=config.TOP_K)
    return {"documents": docs, "trace": [f"🔎 Retrieved {len(docs)} chunks"]}


def grade_documents(state: GraphState):
    docs = state["documents"]
    if not docs:
        return {"trace": ["🧪 Nothing to grade"]}
    # all chunks graded in parallel; a failed grading call keeps the chunk (fail-open)
    results = relevance_chain.batch(
        [{"question": state["question"], "document": d.page_content} for d in docs],
        config={"max_concurrency": len(docs)},
        return_exceptions=True,
    )
    kept = [d for d, r in zip(docs, results)
            if isinstance(r, Exception) or r.relevant == "yes"]
    failed = sum(isinstance(r, Exception) for r in results)
    note = f" ({failed} unreadable grader replies kept)" if failed else ""
    return {"documents": kept,
            "trace": [f"🧪 Kept {len(kept)}/{len(docs)} relevant chunks{note}"]}


def rewrite_query(state: GraphState):
    q = rewrite_chain.invoke({
        "original": state["original_question"],
        "question": state["question"],
    }).content.strip()
    return {"question": q, "retries": state["retries"] + 1,
            "trace": [f"✏️ Rewrote query → “{q}”"]}


def web_search(state: GraphState):
    docs = []
    try:
        from tavily import TavilyClient
        res = TavilyClient(api_key=config.TAVILY_API_KEY).search(
            state["original_question"], max_results=3
        )
        docs = [Document(page_content=r["content"], metadata={"source": r["url"]})
                for r in res.get("results", [])]
    except Exception as e:
        return {"used_web": True,
                "trace": [f"🌐 Web search unavailable ({type(e).__name__})"]}
    return {"documents": docs, "used_web": True,
            "trace": [f"🌐 Web fallback: {len(docs)} results"]}


def generate(state: GraphState):
    n = state["gen_retries"] + 1
    if not state["documents"]:
        return {"generation": "I couldn't find this in your documents.",
                "gen_retries": n,
                "trace": ["⚠️ No relevant context found"]}
    strict = (" Your previous answer had unsupported claims. "
              "Use ONLY facts stated in the context."
              if state["gen_retries"] > 0 else "")
    out = generate_chain.invoke({
        "context": _format_docs(state["documents"]),
        "question": state["original_question"],
        "strict": strict,
    }).content
    out = (out or "").strip() or "I couldn't produce an answer from the retrieved context."
    return {"generation": out, "gen_retries": n,
            "trace": [f"✍️ Generated answer (attempt {n})"]}


def check_answer(state: GraphState):
    if not state["documents"] or not state["verify"]:
        why = "no context" if not state["documents"] else "verification off"
        return {"grounded": True, "useful": bool(state["documents"]),
                "trace": [f"⏭️ Check skipped ({why})"]}
    try:
        r = check_chain.invoke({
            "context": _format_docs(state["documents"]),
            "question": state["original_question"],
            "generation": state["generation"],
        })
    except Exception as e:
        return {"grounded": True, "useful": True,
                "trace": [f"⚠️ Verification unavailable ({type(e).__name__}) — answer NOT verified"]}
    g, u = r.grounded == "yes", r.useful == "yes"
    return {"grounded": g, "useful": u,
            "trace": [f"🛡️ Verified → grounded {'✅' if g else '❌'} · useful {'✅' if u else '❌'}"]}


# ---------------- Routing (conditional edges) ----------------
def decide_after_grading(state: GraphState) -> str:
    if state["documents"]:
        return "generate"
    if state["retries"] < state["max_retries"]:
        return "rewrite"
    return "generate" if state["used_web"] else "web_search"


def decide_after_check(state: GraphState) -> str:
    if not state["documents"]:
        return "end"
    if not state["grounded"]:
        return "generate" if state["gen_retries"] <= state["max_retries"] else "end"
    if not state["useful"] and state["retries"] < state["max_retries"]:
        return "rewrite"
    return "end"