"""
Phase 5: Specialist Agents Framework

Three specialized agents for handling different ticket types:
1. Technical Agent - Error resolution, troubleshooting
2. Billing Agent - Payment issues, refunds, subscriptions
3. Resolution Drafter - Multi-agent coordination and response synthesis
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime

from src.rag_retriever import RetrievalContext
from src.orchestrator import TicketAnalysis, RoutingPath
from src.config import LOGGER


class AgentType(Enum):
    """Specialist agent types"""
    TECHNICAL = "technical"
    BILLING = "billing"
    DRAFTER = "drafter"


@dataclass
class AgentResponse:
    """Response from a specialist agent"""
    agent_type: AgentType
    content: str                    # The generated response
    confidence: float               # 0-1 confidence in response
    requires_human_review: bool     # Does this need human verification?
    reasoning: str                  # Why this response was chosen
    tools_used: List[str]          # Which tools were called
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_type': self.agent_type.value,
            'content': self.content,
            'confidence': round(self.confidence, 2),
            'requires_human_review': self.requires_human_review,
            'reasoning': self.reasoning,
            'tools_used': self.tools_used,
            'timestamp': self.timestamp,
        }


@dataclass
class DrafterOutput:
    """Final output from Resolution Drafter"""
    final_response: str
    confidence: float               # 0-1 overall confidence
    sources: List[str]             # Documents/agents used
    agent_contributions: Dict[str, str]  # What each agent contributed
    escalation_recommended: bool    # Should this go to human?
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_response': self.final_response,
            'confidence': round(self.confidence, 2),
            'sources': self.sources,
            'agent_contributions': self.agent_contributions,
            'escalation_recommended': self.escalation_recommended,
            'timestamp': self.timestamp,
        }


class SpecialistAgent(ABC):
    """
    Base class for all specialist agents.
    
    Each agent:
    - Receives a TicketAnalysis from the Orchestrator
    - Uses RAG context and classification to understand the issue
    - May call tools to gather additional information
    - Generates a response with confidence score
    """
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.tools = self._init_tools()
        LOGGER.info(f"{self.agent_type.value} agent initialized")
    
    @abstractmethod
    def _init_tools(self) -> Dict[str, callable]:
        """Initialize tools available to this agent"""
        pass
    
    @abstractmethod
    def process(self, analysis: TicketAnalysis) -> AgentResponse:
        """
        Process a ticket analysis and generate a response.
        
        Args:
            analysis: TicketAnalysis from Orchestrator
            
        Returns:
            AgentResponse with generated content
        """
        pass
    
    def _format_rag_context(self, context: RetrievalContext) -> str:
        """Format RAG context for agent use"""
        if not context.formatted_context:
            return "No relevant documentation found."
        return context.formatted_context
    
    def _calculate_confidence(
        self,
        rag_confidence: float,
        response_quality: float = 0.8,
        tools_used_count: int = 0
    ) -> float:
        """Calculate response confidence from multiple factors"""
        # RAG confidence is primary factor
        rag_factor = rag_confidence * 0.5
        
        # Response quality (agent's internal assessment)
        quality_factor = response_quality * 0.3
        
        # Tools used boost confidence
        tools_factor = min(1.0, tools_used_count * 0.1) * 0.2
        
        total = rag_factor + quality_factor + tools_factor
        return min(1.0, total)
    
    def get_tools(self) -> Dict[str, callable]:
        """Get available tools for this agent"""
        return self.tools


class TechnicalAgent(SpecialistAgent):
    """
    Technical Support Agent
    
    Handles:
    - Error messages and troubleshooting
    - API integration issues
    - Installation and setup problems
    - Bug reports and workarounds
    - Technical documentation queries
    """
    
    def __init__(self):
        super().__init__(AgentType.TECHNICAL)
    
    def _init_tools(self) -> Dict[str, callable]:
        """Initialize technical troubleshooting tools"""
        return {
            'parse_error_message': self._parse_error_message,
            'suggest_workaround': self._suggest_workaround,
            'check_known_issues': self._check_known_issues,
            'retrieve_logs': self._retrieve_logs,
        }
    
    def _parse_error_message(self, error_code: str) -> str:
        """Parse error code and return explanation"""
        # In production, this would query a knowledge base
        error_meanings = {
            '403': 'Forbidden - Access denied',
            '404': 'Not Found - Resource does not exist',
            '500': 'Server Error - Internal server error',
            '503': 'Service Unavailable - Server is down',
        }
        return error_meanings.get(error_code, f"Unknown error code: {error_code}")
    
    def _suggest_workaround(self, issue_type: str) -> str:
        """Suggest temporary workaround for known issues"""
        workarounds = {
            'connection_timeout': 'Try increasing timeout to 30 seconds',
            'rate_limit': 'Wait 1 minute before retrying',
            'cache_issue': 'Clear browser cache and try again',
            'version_conflict': 'Update to latest stable version',
        }
        return workarounds.get(issue_type, "No known workaround available")
    
    def _check_known_issues(self, keywords: List[str]) -> str:
        """Check if this matches any known issues"""
        # In production, would query issue tracker
        return "No known issues match your description"
    
    def _retrieve_logs(self, ticket_id: str) -> str:
        """Retrieve system logs related to the ticket"""
        # In production, would fetch actual logs
        return "Log retrieval would be performed here"
    
    def process(self, analysis: TicketAnalysis) -> AgentResponse:
        """
        Process technical support ticket.
        
        Strategy:
        1. Extract error indicators from ticket content
        2. Use RAG context for documentation
        3. Parse error messages
        4. Suggest solutions with confidence scoring
        """
        LOGGER.info(f"Technical Agent processing: {analysis.ticket_id}")
        
        ticket_content = analysis.content.lower()
        rag_context = self._format_rag_context(analysis.rag_context)
        
        tools_used = []
        response_parts = []
        
        # Step 1: Look for error codes (e.g., 403, 500)
        import re
        error_codes = re.findall(r'\b(4\d{2}|5\d{2})\b', analysis.content)
        if error_codes:
            tools_used.append('parse_error_message')
            for code in error_codes[:1]:  # Use first error code
                error_meaning = self._parse_error_message(code)
                response_parts.append(f"**Error {code}**: {error_meaning}")
        
        # Step 2: Check for common technical issues
        issue_keywords = ['timeout', 'rate limit', 'cache', 'version']
        matched_issues = [kw for kw in issue_keywords if kw in ticket_content]
        if matched_issues:
            tools_used.append('suggest_workaround')
            for issue in matched_issues[:1]:
                workaround = self._suggest_workaround(issue.replace(' ', '_'))
                response_parts.append(f"\n**Workaround**: {workaround}")
        
        # Step 3: Add RAG context
        if analysis.rag_context.total_documents > 0:
            response_parts.append(f"\n**Documentation**:\n{rag_context[:500]}")
        
        # Step 4: Provide next steps
        response_parts.append(
            "\n**Next Steps**:\n"
            "1. Try the suggested workaround\n"
            "2. Check the documentation above\n"
            "3. If issue persists, contact support with error logs"
        )
        
        # Combine response
        full_response = "".join(response_parts)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            analysis.rag_context.confidence_score,
            response_quality=0.85 if error_codes else 0.7,
            tools_used_count=len(tools_used)
        )
        
        return AgentResponse(
            agent_type=AgentType.TECHNICAL,
            content=full_response,
            confidence=confidence,
            requires_human_review=confidence < 0.6,
            reasoning=f"Used {len(tools_used)} tools, RAG confidence {analysis.rag_context.confidence_score:.1%}",
            tools_used=tools_used
        )


class BillingAgent(SpecialistAgent):
    """
    Billing & Payment Agent
    
    Handles:
    - Charging and billing issues
    - Refund requests
    - Subscription management
    - Invoice questions
    - Payment method issues
    """
    
    def __init__(self):
        super().__init__(AgentType.BILLING)
    
    def _init_tools(self) -> Dict[str, callable]:
        """Initialize billing tools"""
        return {
            'check_transaction': self._check_transaction,
            'process_refund': self._process_refund,
            'modify_subscription': self._modify_subscription,
            'generate_invoice': self._generate_invoice,
        }
    
    def _check_transaction(self, keywords: List[str]) -> str:
        """Check transaction history"""
        return "Transaction verification would be performed here"
    
    def _process_refund(self, reason: str) -> str:
        """Process refund request"""
        return f"Refund for '{reason}' would be initiated (requires verification)"
    
    def _modify_subscription(self, action: str) -> str:
        """Modify subscription (upgrade/downgrade/cancel)"""
        actions = {
            'upgrade': 'Your subscription can be upgraded at any time',
            'downgrade': 'Downgrade takes effect on next billing cycle',
            'cancel': 'Your subscription will be cancelled immediately',
        }
        return actions.get(action, "Subscription modification would be processed")
    
    def _generate_invoice(self, period: str) -> str:
        """Generate invoice for billing period"""
        return f"Invoice for {period} would be generated"
    
    def process(self, analysis: TicketAnalysis) -> AgentResponse:
        """
        Process billing ticket.
        
        Strategy:
        1. Identify billing issue type
        2. Check transaction history (if applicable)
        3. Provide policy information from RAG
        4. Suggest resolution
        """
        LOGGER.info(f"Billing Agent processing: {analysis.ticket_id}")
        
        ticket_content = analysis.content.lower()
        rag_context = self._format_rag_context(analysis.rag_context)
        
        tools_used = []
        response_parts = []
        
        # Step 1: Identify billing issue type
        issue_type = None
        if 'charged' in ticket_content or 'charge' in ticket_content:
            issue_type = 'duplicate_charge'
            tools_used.append('check_transaction')
        elif 'refund' in ticket_content:
            issue_type = 'refund_request'
            tools_used.append('process_refund')
        elif 'subscription' in ticket_content:
            issue_type = 'subscription'
            tools_used.append('modify_subscription')
        
        # Step 2: Address the issue
        if issue_type == 'duplicate_charge':
            response_parts.append(
                "**Duplicate Charge Issue**\n\n"
                "I understand you were charged multiple times. Here's what we'll do:\n"
                "1. Verify the duplicate charge in our system\n"
                "2. Process a refund for the duplicate amount\n"
                "3. Add a credit to your account as compensation"
            )
        elif issue_type == 'refund_request':
            response_parts.append(
                "**Refund Request**\n\n"
                "We can help with your refund. Our policy:\n"
                "- Full refund within 30 days of purchase\n"
                "- Partial refund after 30 days (prorated)\n"
                "- Processing time: 3-5 business days"
            )
        elif issue_type == 'subscription':
            response_parts.append(
                "**Subscription Management**\n\n"
                "You can manage your subscription at any time:\n"
                "- Upgrade anytime (changes effective immediately)\n"
                "- Downgrade (takes effect on next cycle)\n"
                "- Cancel (effective immediately)"
            )
        else:
            response_parts.append(
                "**Billing Inquiry**\n\n"
                "Thank you for contacting us about your billing.\n"
                "I've reviewed your account and we can help resolve this."
            )
        
        # Step 3: Add policy information
        if analysis.rag_context.total_documents > 0:
            response_parts.append(f"\n**Policy Information**:\n{rag_context[:400]}")
        
        # Step 4: Next steps
        response_parts.append(
            "\n**Next Steps**:\n"
            "1. Reply to confirm the issue\n"
            "2. Provide any order numbers if available\n"
            "3. Our billing team will process within 24 hours"
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            analysis.rag_context.confidence_score,
            response_quality=0.9 if issue_type else 0.75,
            tools_used_count=len(tools_used)
        )
        
        return AgentResponse(
            agent_type=AgentType.BILLING,
            content="".join(response_parts),
            confidence=confidence,
            requires_human_review=confidence < 0.65,
            reasoning=f"Issue type: {issue_type}, Used {len(tools_used)} tools",
            tools_used=tools_used
        )


class ResolutionDrafter(SpecialistAgent):
    """
    Resolution Drafter - Multi-Agent Coordinator
    
    Responsibilities:
    - Receives responses from specialist agents
    - Coordinates between agents for complex issues
    - Synthesizes final response
    - Ensures quality and coherence
    - Recommends escalation if needed
    """
    
    def __init__(self):
        super().__init__(AgentType.DRAFTER)
    
    def _init_tools(self) -> Dict[str, callable]:
        """Initialize drafting tools"""
        return {
            'synthesize_responses': self._synthesize_responses,
            'check_response_quality': self._check_response_quality,
            'recommend_escalation': self._recommend_escalation,
        }
    
    def _synthesize_responses(self, responses: List[AgentResponse]) -> str:
        """Synthesize multiple agent responses into one"""
        if not responses:
            return "No agent responses available"
        
        if len(responses) == 1:
            return responses[0].content
        
        # Multiple responses - synthesize them
        parts = ["Based on analysis from our specialist teams:\n"]
        for response in responses:
            parts.append(f"\n**{response.agent_type.value.title()} Analysis**:\n{response.content}\n")
        
        return "".join(parts)
    
    def _check_response_quality(self, response: str, confidence: float) -> Dict[str, Any]:
        """Check if response meets quality standards"""
        return {
            'meets_quality': confidence > 0.6,
            'clarity_score': 0.8,
            'completeness_score': 0.75,
            'professional_tone': True,
        }
    
    def _recommend_escalation(self, confidence: float, complexity: int) -> bool:
        """Recommend escalation to human agent"""
        # Escalate if low confidence or high complexity
        return confidence < 0.5 or complexity > 2
    
    def draft_response(
        self,
        analysis: TicketAnalysis,
        agent_responses: List[AgentResponse]
    ) -> DrafterOutput:
        """
        Draft final response from specialist agent inputs.
        
        Args:
            analysis: Original ticket analysis
            agent_responses: Responses from specialist agents
            
        Returns:
            Final DrafterOutput with synthesized response
        """
        LOGGER.info(f"Resolution Drafter processing: {analysis.ticket_id}")
        
        # Synthesize agent responses
        synthesized = self._synthesize_responses(agent_responses)
        
        # Determine overall confidence
        if agent_responses:
            avg_confidence = sum(r.confidence for r in agent_responses) / len(agent_responses)
        else:
            avg_confidence = analysis.rag_context.confidence_score
        
        # Check if escalation needed
        escalate = any(r.requires_human_review for r in agent_responses)
        escalate = escalate or avg_confidence < 0.6
        
        # Build sources list
        sources = [
            d.source for d in analysis.rag_context.retrieved_documents
        ]
        
        # Build agent contributions
        contributions = {
            r.agent_type.value: r.reasoning for r in agent_responses
        }
        
        # Add quality note to response
        final_response = synthesized
        if escalate:
            final_response += (
                "\n\n⚠️ **Note**: This response contains complex elements and has been flagged for quality review. "
                "A human specialist will verify before sending."
            )
        
        return DrafterOutput(
            final_response=final_response,
            confidence=avg_confidence,
            sources=sources,
            agent_contributions=contributions,
            escalation_recommended=escalate
        )
    
    def process(self, analysis: TicketAnalysis) -> AgentResponse:
        """
        Drafting agent doesn't directly process tickets;
        instead it orchestrates other agents.
        This method is kept for compatibility.
        """
        return AgentResponse(
            agent_type=AgentType.DRAFTER,
            content="Resolution Drafter is a coordinator, not a primary responder",
            confidence=1.0,
            requires_human_review=False,
            reasoning="Drafting agent coordinates other specialists",
            tools_used=[]
        )


class SpecialistAgentManager:
    """
    Manager for coordinating all specialist agents.
    
    Handles:
    - Agent selection based on ticket type
    - Agent collaboration for complex issues
    - Response synthesis
    - Final quality check
    """
    
    def __init__(self):
        self.technical_agent = TechnicalAgent()
        self.billing_agent = BillingAgent()
        self.drafter = ResolutionDrafter()
        LOGGER.info("SpecialistAgentManager initialized with all agents")
    
    def process_simple_ticket(
        self,
        analysis: TicketAnalysis
    ) -> AgentResponse:
        """
        Process simple ticket with single specialist agent.
        
        Routes to appropriate agent based on category.
        """
        LOGGER.info(f"Processing simple ticket: {analysis.ticket_id}")
        
        category = analysis.classification.category.value
        
        if category == 'billing':
            return self.billing_agent.process(analysis)
        else:  # Default to technical
            return self.technical_agent.process(analysis)
    
    def process_complex_ticket(
        self,
        analysis: TicketAnalysis
    ) -> DrafterOutput:
        """
        Process complex ticket with multiple specialist agents.
        
        Gets responses from all relevant agents and synthesizes.
        """
        LOGGER.info(f"Processing complex ticket: {analysis.ticket_id}")
        
        # Get responses from both specialists
        technical_response = self.technical_agent.process(analysis)
        billing_response = self.billing_agent.process(analysis)
        
        agent_responses = [technical_response, billing_response]
        
        # Synthesize final response
        final_output = self.drafter.draft_response(analysis, agent_responses)
        
        return final_output
    
    def route_ticket(
        self,
        analysis: TicketAnalysis
    ) -> Dict[str, Any]:
        """
        Route ticket to appropriate processing path.
        
        Uses orchestrator routing decision to determine
        which agents should handle the ticket.
        """
        routing_path = analysis.routing_decision.path
        
        if routing_path == RoutingPath.AUTO_RESPOND:
            # Return RAG context directly
            return {
                'type': 'auto_response',
                'content': analysis.rag_context.formatted_context,
                'confidence': analysis.rag_context.confidence_score,
            }
        
        elif routing_path == RoutingPath.SINGLE_SPECIALIST:
            # Use single specialist agent
            response = self.process_simple_ticket(analysis)
            return {
                'type': 'specialist_response',
                'response': response.to_dict(),
            }
        
        else:  # SPECIALIST_TEAM
            # Use multiple agents coordinated by drafter
            output = self.process_complex_ticket(analysis)
            return {
                'type': 'team_response',
                'output': output.to_dict(),
            }


# Example usage
if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    
    # Test agent creation
    tech_agent = TechnicalAgent()
    billing_agent = BillingAgent()
    drafter = ResolutionDrafter()
    
    print("✓ All specialist agents initialized successfully")
