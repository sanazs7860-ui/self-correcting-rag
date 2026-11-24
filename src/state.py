import operator
from typing import Annotated, List, Literal, TypedDict
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    original_question: str
    question: str
    documents: List[Document]
    generation: str
    retries: int
    gen_retries: int
    max_retries: int          # set from the UI
    verify: bool              # run the hallucination check?
    used_web: bool
    grounded: bool
    useful: bool
    trace: Annotated[List[str], operator.add]


class GradeRelevance(BaseModel):
    """Is this chunk relevant to the question?"""
    relevant: Literal["yes", "no"] = Field(description="'yes' if chunk helps answer the question")


class AnswerCheck(BaseModel):
    """Combined hallucination + usefulness check (one LLM call instead of two)."""
    grounded: Literal["yes", "no"] = Field(description="'yes' if every claim is supported by the context")
    useful: Literal["yes", "no"] = Field(description="'yes' if the answer resolves the question")
