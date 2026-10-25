import logging
from typing import Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

Route = Literal["greeting", "general_knowledge", "document", "uncertain"]


class QuestionPlan(BaseModel):
    """The AI's decision about one user message. Descriptions are shown to the AI."""

    route: Route = Field(
        description=(
            "greeting: hello, thanks, small talk. "
            "general_knowledge: answerable without any uploaded document. "
            "document: needs facts from the user's uploaded documents. "
            "uncertain: cannot tell."
        )
    )
    search_question: str = Field(
        description=(
            "The latest message rewritten as ONE complete, standalone question for searching "
            "documents: follow-ups resolved from the earlier conversation without adding new "
            "details, everyday terms followed by the likely formal term in brackets. For "
            "greetings and general questions repeat it as is."
        )
    )
    reason: str = Field(description="One short sentence explaining the route.")


PLANNER_SYSTEM_PROMPT = """You are the routing step of a chatbot that answers questions about \
PDF documents the user uploaded (contracts, policies, reports, manuals, notes...), and also \
chats about general topics.

For the user's latest message decide the route:
- greeting: hello, thanks, goodbye, small talk.
- general_knowledge: a question that does not depend on the uploaded documents \
(e.g. "What is Python?").
- document: a question whose answer would come from the uploaded documents \
(facts, figures, clauses, deadlines, summaries).
- uncertain: you genuinely cannot tell.

If the message could plausibly be about facts, figures, policies or details from the user's \
documents, prefer document or uncertain over general_knowledge. Questions about the user's own \
situation ("my marks", "my leave", "can I...", "what happens if I...") or about rules, fees, \
deadlines and procedures are almost always document questions. Choose general_knowledge only \
when the answer is public knowledge that no particular organisation's document would change.

Also write search_question: the latest message as one complete standalone question for searching \
the documents. Rules for it:
1. Use the earlier conversation only to work out what words like "it", "that" or "and for hostel?" \
refer to. Carry over the TOPIC (for example "library fines"), but never copy a section or clause \
number, name, date or figure from the earlier conversation unless the new message itself mentions \
it. Narrowing the question this way would hide the right answer.
2. Documents use formal wording and users use everyday wording. When the user's words are everyday \
terms, keep them AND add the likely formal term in brackets, for example "money back" becomes \
"refund (money back)" and "fired" becomes "termination of employment (fired)".
3. Keep numbers, names and exact terms unchanged.
4. For greetings and general questions, repeat the message as it is."""


class QuestionPlanner(Protocol):
    def plan(self, question: str, history: list[dict]) -> QuestionPlan: ...


class GroqQuestionPlanner:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def plan(self, question: str, history: list[dict]) -> QuestionPlan:
        model = self._llm.model(0.0).with_structured_output(QuestionPlan)
        recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        user_text = (
            f"Earlier conversation (may be empty):\n{recent or '(none)'}\n\n"
            f"Latest user message:\n{question}"
        )
        try:
            result = model.invoke([SystemMessage(PLANNER_SYSTEM_PROMPT), HumanMessage(user_text)])
            plan = result if isinstance(result, QuestionPlan) else QuestionPlan.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - the library raises many different error types
            # Safe default: "uncertain" still searches the documents, using the original words.
            logger.warning("Planner failed (%s); using 'uncertain'", type(exc).__name__)
            return QuestionPlan(route="uncertain", search_question=question, reason="planner unavailable")
        if not plan.search_question.strip():
            plan = plan.model_copy(update={"search_question": question})
        logger.info("Question routed as %s", plan.route)
        return plan
