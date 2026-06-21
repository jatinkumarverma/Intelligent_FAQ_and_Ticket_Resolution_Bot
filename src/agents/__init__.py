"""
src/agents - Specialist Agent Implementations

This package contains specialized agents that handle different types of support tickets:

- TechnicalAgent: Handles technical issues, errors, and troubleshooting
- BillingAgent: Handles payment, refund, and subscription issues
- ResolutionDrafter: Coordinates multiple agents and synthesizes final responses

Phase 5: Specialist Agents Implementation
"""

from .specialist_agents import (
    TechnicalAgent,
    BillingAgent,
    ResolutionDrafter,
    SpecialistAgentManager,
    SpecialistAgent,
    AgentType,
    AgentResponse,
    DrafterOutput,
)

__all__ = [
    'TechnicalAgent',
    'BillingAgent',
    'ResolutionDrafter',
    'SpecialistAgentManager',
    'SpecialistAgent',
    'AgentType',
    'AgentResponse',
    'DrafterOutput',
]
