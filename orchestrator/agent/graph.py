"""
The agent's decision-making core, built with LangGraph.

Flow:
    route -> (rag | web_search | direct) -> synthesize -> END

- route: LLM call that decides which tool (if any) this question needs.
- rag / web_search / direct: execute the chosen path, populate tool_output.
- synthesize: turns the question + tool_output into a short, voice-friendly
  final answer.

Why LangGraph instead of just if/else in Python: this is a genuinely simple
graph right now, but it's built to extend — e.g. Phase 4+ could add a
critic node after synthesize that checks answer quality and loops back to
route with a reformulated query. That's why state is a typed dict flowing
through named nodes rather than a plain function call chain — it's the same
pattern your Research Assistant project uses, on purpose.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from groq import Groq

from orchestrator.config import config
from orchestrator.agent.tools.rag_tool import RAGTool
from orchestrator.agent.tools.web_search_tool import WebSearchTool

groq_client = Groq(api_key=config.GROQ_API_KEY)


class AgentState(TypedDict):
    question: str
    tool_choice: str       # "rag" | "web_search" | "direct"
    tool_output: str        # retrieved context, empty string if direct
    sources: list           # doc IDs or URLs, empty if direct
    final_answer: str


ROUTER_PROMPT = """You are a routing component in a voice agent. Decide which \
tool is needed to answer the user's question. Respond with EXACTLY ONE WORD, \
nothing else:

- "rag" if the question is about internal company policy, HR, onboarding, \
expenses, leave, or engineering processes/runbooks — anything an internal \
knowledge base would cover.
- "web_search" if the question needs current, real-world, or general \
knowledge information not specific to an internal company (news, facts, \
public information, anything time-sensitive).
- "direct" if the question is small talk, a simple request, or something \
you can answer confidently without looking anything up.

Question: {question}

Answer with one word only: rag, web_search, or direct."""

SYNTHESIS_PROMPT_WITH_CONTEXT = """You are a helpful voice assistant speaking \
your answer aloud. Use the retrieved information below to answer the \
question naturally and conversationally. Keep it short — 2-3 sentences max. \
Don't mention "the context" or "the documents" or "according to the search \
results" — just answer naturally like you already knew it.

Retrieved information:
{context}

Question: {question}

Your spoken answer:"""

SYNTHESIS_PROMPT_DIRECT = """You are a helpful voice assistant speaking your \
answer aloud. Keep answers short and conversational — 2-3 sentences max, no \
lists or markdown, since this will be read aloud by text-to-speech.

Question: {question}

Your spoken answer:"""


def route_node(state: AgentState) -> AgentState:
    completion = groq_client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=state["question"])}],
        temperature=0,
        max_tokens=10,
    )
    raw_choice = completion.choices[0].message.content.strip().lower()

    # Defensive parsing — LLMs sometimes add punctuation or extra words
    # despite instructions. Fall back to "direct" if we don't recognize it,
    # since that's the safest failure mode (no tool call rather than a
    # crash on an unrecognized tool name).
    if "rag" in raw_choice:
        tool_choice = "rag"
    elif "web_search" in raw_choice or "web search" in raw_choice:
        tool_choice = "web_search"
    else:
        tool_choice = "direct"

    return {**state, "tool_choice": tool_choice}


def rag_node(state: AgentState) -> AgentState:
    rag_tool = RAGTool()
    result = rag_tool.query(state["question"])
    return {**state, "tool_output": result["context"], "sources": result["sources"]}


def web_search_node(state: AgentState) -> AgentState:
    web_tool = WebSearchTool()
    result = web_tool.query(state["question"])
    return {**state, "tool_output": result["context"], "sources": result["sources"]}


def direct_node(state: AgentState) -> AgentState:
    return {**state, "tool_output": "", "sources": []}


def synthesize_node(state: AgentState) -> AgentState:
    if state["tool_output"]:
        prompt = SYNTHESIS_PROMPT_WITH_CONTEXT.format(
            context=state["tool_output"], question=state["question"]
        )
    else:
        prompt = SYNTHESIS_PROMPT_DIRECT.format(question=state["question"])

    completion = groq_client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    answer = completion.choices[0].message.content.strip()
    return {**state, "final_answer": answer}


def route_decision(state: AgentState) -> Literal["rag", "web_search", "direct"]:
    return state["tool_choice"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("direct", direct_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        route_decision,
        {"rag": "rag", "web_search": "web_search", "direct": "direct"},
    )
    graph.add_edge("rag", "synthesize")
    graph.add_edge("web_search", "synthesize")
    graph.add_edge("direct", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


class VoiceAgent:
    """Thin wrapper so the rest of the codebase doesn't need to know about
    LangGraph internals — just call .run(question)."""

    def __init__(self):
        self.graph = build_graph()

    def run(self, question: str) -> dict:
        """Returns the full final state, including tool_choice and sources —
        useful for debugging/eval, not just the final_answer string."""
        result = self.graph.invoke(
            {
                "question": question,
                "tool_choice": "",
                "tool_output": "",
                "sources": [],
                "final_answer": "",
            }
        )
        return result
