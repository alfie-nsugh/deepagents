"""ask_human tool for Human-in-the-Loop Questions.

This module provides tools and middleware for agents to ask humans questions
when they're stuck, uncertain, or need guidance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.messages import AIMessage
from langchain.tools import BaseTool, tool
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


class QuestionPriority(str, Enum):
    """Priority level for questions."""
    
    BLOCKING = "blocking"  # Agent cannot proceed without answer
    HIGH = "high"  # Important but agent can attempt to continue
    MEDIUM = "medium"  # Helpful but not critical
    NICE_TO_HAVE = "nice_to_have"  # Optional clarification


class QuestionOption(BaseModel):
    """Option for multiple choice questions."""
    
    id: str = Field(description="Unique ID for this option")
    label: str = Field(description="Short label for this option")
    description: str | None = Field(default=None, description="Optional longer description")


class Question(BaseModel):
    """A question for the human."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(description="The question text")
    priority: QuestionPriority = Field(
        default=QuestionPriority.MEDIUM,
        description="How urgent is this question?"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent's confidence level (0-1). Lower = more uncertain."
    )
    options: list[QuestionOption] | None = Field(
        default=None,
        description="Options for multiple choice questions"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (file, line number, branch, etc.)"
    )
    subject: str | None = Field(
        default=None,
        description="Category/subject for grouping questions"
    )


class AskHumanInput(BaseModel):
    """Input schema for the ask_human tool."""
    
    question: str = Field(
        description="The question to ask the human. Be specific and clear."
    )
    priority: Literal["blocking", "high", "medium", "nice_to_have"] = Field(
        default="medium",
        description="How urgent is this question? blocking=cannot proceed, high=important, medium=helpful, nice_to_have=optional"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Your confidence level about the current approach (0=completely stuck, 1=just confirming)"
    )
    options: list[str] | None = Field(
        default=None,
        description="Optional list of choices for multiple choice questions"
    )
    context_file: str | None = Field(
        default=None,
        description="File path related to this question"
    )
    subject: str | None = Field(
        default=None,
        description="Category for grouping (e.g., 'API Design', 'Testing Strategy')"
    )


def create_ask_human_tool() -> BaseTool:
    """Create the ask_human tool that triggers an interrupt.
    
    This tool uses LangGraph's interrupt mechanism to pause execution
    and wait for a human response.
    """
    
    @tool("ask_human", args_schema=AskHumanInput)
    def ask_human(
        question: str,
        priority: Literal["blocking", "high", "medium", "nice_to_have"] = "medium",
        confidence: float = 0.5,
        options: list[str] | None = None,
        context_file: str | None = None,
        subject: str | None = None,
    ) -> str:
        """Ask the human a question and wait for their response.
        
        Use this tool primarily when you:
        - Are stuck and need guidance
        - Are uncertain about the right approach

        Less importantly, use it when you:
        - Need clarification on requirements
        - Want to confirm before making significant changes
        
        The human will see your question with context and can provide
        a text answer or choose from options (if provided).
        
        Args:
            question: Your question for the human. Be specific and clear.
            priority: How urgent is this? 
                     - "blocking": Cannot proceed without answer
                     - "high": Important, but can attempt to continue
                     - "medium": Helpful clarification
                     - "nice_to_have": Optional, just confirming
            confidence: Your confidence level (0-1). Lower = more uncertain.
            options: Optional list of choices for multiple choice.
            context_file: File path this question relates to.
            subject: Category for grouping similar questions.
        
        Returns:
            The human's response to your question.
        """
        # Build question options if provided
        question_options = None
        if options:
            question_options = [
                QuestionOption(id=str(i), label=opt)
                for i, opt in enumerate(options)
            ]
        
        # Build the question object
        q = Question(
            text=question,
            priority=QuestionPriority(priority),
            confidence=confidence,
            options=question_options,
            context={"file": context_file} if context_file else {},
            subject=subject,
        )
        
        # Interrupt and wait for human response
        # The UI will display this question and collect the answer
        response = interrupt({
            "type": "ask_human",
            "questions": [q.model_dump()],
        })
        
        # Response format from UI: {"answers": {"question_id": "answer"}, "type": "questions_answered"}
        if isinstance(response, dict):
            answers = response.get("answers", {})
            if q.id in answers:
                return answers[q.id]
            # Fallback: return first answer if only one question
            if answers:
                return next(iter(answers.values()))
        
        # Raw string response
        if isinstance(response, str):
            return response
        
        return str(response)
    
    return ask_human


@dataclass
class QuestionsMiddleware(AgentMiddleware):
    """Middleware that collects questions and manages the Q&A flow.
    
    This middleware:
    1. Injects the ask_human tool into the agent
    2. Batches multiple questions if agent asks several in quick succession
    3. Tracks questions and answers for trajectory collection
    """
    
    questions_asked: list[Question] = field(default_factory=list)
    answers_received: dict[str, str] = field(default_factory=dict)
    
    def modify_state(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> AgentState:
        """Add ask_human tool to the agent's available tools."""
        ask_human_tool = create_ask_human_tool()
        
        # Add to tools if not already present
        existing_tools = state.get("tools", [])
        tool_names = {t.name for t in existing_tools if hasattr(t, "name")}
        
        if "ask_human" not in tool_names:
            state["tools"] = list(existing_tools) + [ask_human_tool]
        
        return state
    
    def process_response(
        self,
        state: AgentState,
        response: AIMessage,
        runtime: Runtime,
    ) -> AIMessage:
        """Track questions asked by the agent."""
        # Check if any tool calls are ask_human
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.get("name") == "ask_human":
                    args = tc.get("args", {})
                    q = Question(
                        text=args.get("question", ""),
                        priority=QuestionPriority(args.get("priority", "medium")),
                        confidence=args.get("confidence", 0.5),
                        subject=args.get("subject"),
                    )
                    self.questions_asked.append(q)
        
        return response
    
    def get_trajectory_data(self) -> dict[str, Any]:
        """Return question/answer data for trajectory collection."""
        return {
            "questions": [q.model_dump() for q in self.questions_asked],
            "answers": self.answers_received,
            "total_questions": len(self.questions_asked),
            "blocking_questions": sum(
                1 for q in self.questions_asked 
                if q.priority == QuestionPriority.BLOCKING
            ),
        }


# Convenience function to add questions support to an existing agent
def add_questions_support(
    agent_middleware: list[AgentMiddleware],
) -> tuple[list[AgentMiddleware], QuestionsMiddleware]:
    """Add questions support to an agent's middleware stack.
    
    Args:
        agent_middleware: Existing middleware list
        
    Returns:
        Tuple of (updated middleware list, QuestionsMiddleware instance)
    """
    questions_middleware = QuestionsMiddleware()
    return [questions_middleware] + list(agent_middleware), questions_middleware
