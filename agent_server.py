"""Agent Server for deep-agents-ui.

This module creates a LangGraph server that can be connected to from deep-agents-ui.
It uses create_cli_agent from deepagents-cli and exposes it as a deployable graph.

Usage:
    1. Set environment variables (or use a .env file):
        - LANGSMITH_API_KEY (optional, for tracing)
        - GOOGLE_API_KEY (for Gemini models)
        - ANTHROPIC_API_KEY (for Claude models)
        - OPENAI_API_KEY (for OpenAI models)
    
    2. Run with langgraph CLI:
        langgraph dev --config langgraph.json
    
    3. Configure deep-agents-ui:
        - Deployment URL: http://localhost:8123
        - Assistant ID: deep-agent (or your custom name)
        - LangSmith API Key: (optional)

Example langgraph.json:
    {
        "graphs": {
            "deep-agent": "./agent_server.py:create_server_agent"
        },
        "host": "0.0.0.0",
        "port": 8123
    }
"""

import os

from deepagents.middleware.questions import create_ask_human_tool
from deepagents_cli.agent import create_cli_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph


# ============================================
# Configuration - Edit these for your setup
# ============================================

# Assistant ID - this is the name shown in deep-agents-ui
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "deep-agent")

# Model configuration (format: "provider:model_name")
# Examples: "google_genai:gemini-3-flash-preview", "anthropic:claude-sonnet-4-5-20250929", "openai:gpt-4o"
MODEL_NAME = os.getenv("MODEL_NAME", "google_genai:gemini-3-flash-preview")

# Enable/disable features
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
ENABLE_SKILLS = os.getenv("ENABLE_SKILLS", "true").lower() == "true"
ENABLE_SHELL = os.getenv("ENABLE_SHELL", "true").lower() == "true"
ENABLE_QUESTIONS = os.getenv("ENABLE_QUESTIONS", "true").lower() == "true"

# Auto-approve mode (bypass HITL for all tools)
AUTO_APPROVE = os.getenv("AUTO_APPROVE", "false").lower() == "true"

# Postgres connection for production persistence (optional)
POSTGRES_URI = os.getenv("POSTGRES_URI", None)


def create_server_agent(
    assistant_id: str | None = None,
    model: str | None = None,
    auto_approve: bool | None = None,
    enable_questions: bool | None = None,
) -> CompiledStateGraph:
    """Create an agent for the LangGraph server.
    
    This function is the entry point called by LangGraph when starting the server.
    It creates a fully configured agent with HITL support including the ask_human tool.
    
    Args:
        assistant_id: Override the default assistant ID
        model: Override the default model
        auto_approve: Override auto-approve setting
        enable_questions: Override questions feature
        
    Returns:
        A compiled LangGraph agent ready for deployment
    """
    # Use provided values or fall back to defaults
    _assistant_id = assistant_id or ASSISTANT_ID
    _model = model or MODEL_NAME
    _auto_approve = auto_approve if auto_approve is not None else AUTO_APPROVE
    _enable_questions = enable_questions if enable_questions is not None else ENABLE_QUESTIONS
    
    # Initialize the model object from the string
    model_obj = init_chat_model(_model)
    
    # Build list of additional tools
    extra_tools = []
    if _enable_questions:
        ask_human_tool = create_ask_human_tool()
        extra_tools.append(ask_human_tool)
        print(f"✅ ask_human tool enabled")
    
    # Create the base agent using create_cli_agent
    # Note: use_persistence=False because LangGraph API handles persistence automatically
    agent, backend = create_cli_agent(
        model=model_obj,
        assistant_id=_assistant_id,
        tools=extra_tools,  # Pass ask_human tool
        auto_approve=_auto_approve,
        enable_memory=ENABLE_MEMORY,
        enable_skills=ENABLE_SKILLS,
        enable_shell=ENABLE_SHELL,
        use_persistence=False,  # LangGraph API handles persistence
    )
    
    print(f"✅ Agent server created")
    print(f"   Assistant ID: {_assistant_id}")
    print(f"   Model: {_model}")
    print(f"   Questions enabled: {_enable_questions}")
    
    return agent


def create_checkpointer():
    """Create the appropriate checkpointer based on configuration."""
    if POSTGRES_URI:
        print(f"Using PostgreSQL checkpointer")
        return PostgresSaver.from_conn_string(POSTGRES_URI)
    else:
        print(f"Using in-memory checkpointer (state will not persist across restarts)")
        return InMemorySaver()


# ============================================
# LangGraph Server Entry Point
# ============================================

# This is what LangGraph CLI uses when you run `langgraph dev`
graph = create_server_agent()


# Alternative: Export factory function for programmatic use
def get_graph() -> CompiledStateGraph:
    """Get the agent graph for programmatic use."""
    return create_server_agent()


if __name__ == "__main__":
    # Quick test - print agent info
    print("\n" + "=" * 60)
    print("DeepAgents Server Configuration")
    print("=" * 60)
    print(f"Assistant ID:     {ASSISTANT_ID}")
    print(f"Model:            {MODEL_NAME}")
    print(f"Enable Memory:    {ENABLE_MEMORY}")
    print(f"Enable Skills:    {ENABLE_SKILLS}")
    print(f"Enable Shell:     {ENABLE_SHELL}")
    print(f"Enable Questions: {ENABLE_QUESTIONS}")
    print(f"Auto Approve:     {AUTO_APPROVE}")
    print(f"Postgres URI:     {'Set' if POSTGRES_URI else 'Not set (using memory)'}")
    print("=" * 60)
    print("\nTo start the server, run:")
    print("  langgraph dev")
    print("\nOr for production:")
    print("  langgraph up")
    print("\nThen configure deep-agents-ui with:")
    print(f"  Deployment URL: http://localhost:8123")
    print(f"  Assistant ID:   {ASSISTANT_ID}")
    print("=" * 60 + "\n")
