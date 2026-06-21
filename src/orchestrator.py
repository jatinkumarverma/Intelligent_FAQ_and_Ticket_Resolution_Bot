"""
Phase 4: Orchestrator Agent

Main routing agent that:
1. Receives incoming tickets
2. Classifies tickets by type and complexity
3. Performs RAG retrieval with confidence scoring
4. Routes to appropriate resolution path:
   - High confidence (>0.85): Direct response
   - Medium confidence (0.5-0.85): Single specialist agent
   - Low confidence (<0.5): Escalate to specialist team
5. Coordinates specialist agents for complex issues
6. Synthesizes final response
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.rag_retriever import RAGRetriever, RetrievalContext
from src.vector_store import VectorStore
from src.config import (
    RAG_K_RETRIEVAL,
    RAG_SIMILARITY_THRESHOLD,
    RAG_RELEVANCE_THRESHOLD,
    LOGGER,
)


class TicketSeverity(Enum):
    """Ticket severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCategory(Enum):
    """Supported ticket categories"""
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    GENERAL = "general"
    FEATURE = "feature"
    BUG = "bug"
    UNKNOWN = "unknown"


class RoutingPath(Enum):
    """Routing decision paths"""
    AUTO_RESPOND = "auto_respond"  # High confidence - direct response
    SINGLE_SPECIALIST = "single_specialist"  # Medium confidence - one agent
    SPECIALIST_TEAM = "specialist_team"  # Low confidence - team escalation
    MANUAL_REVIEW = "manual_review"  # Cannot resolve


@dataclass
class ClassificationResult:
    """Ticket classification output"""
    category: TicketCategory
    severity: TicketSeverity
    keywords: List[str]
    confidence: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'keywords': self.keywords,
            'confidence': round(self.confidence, 2),
        }


@dataclass
class RoutingDecision:
    """Routing decision output"""
    path: RoutingPath
    confidence_score: float  # 0-1, from RAG
    recommended_specialist: Optional[str] = None  # Which agent to route to
    reason: str = ""
    escalation_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path.value,
            'confidence_score': round(self.confidence_score, 2),
            'recommended_specialist': self.recommended_specialist,
            'reason': self.reason,
            'escalation_reason': self.escalation_reason,
        }


@dataclass
class TicketAnalysis:
    """Complete ticket analysis"""
    ticket_id: str
    content: str
    classification: ClassificationResult
    rag_context: RetrievalContext
    routing_decision: RoutingDecision
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticket_id': self.ticket_id,
            'content': self.content,
            'classification': self.classification.to_dict(),
            'rag_context': {
                'original_query': self.rag_context.original_query,
                'confidence': round(self.rag_context.confidence_score, 2),
                'documents_count': self.rag_context.total_documents,
                'retrieval_method': self.rag_context.retrieval_method,
            },
            'routing_decision': self.routing_decision.to_dict(),
            'timestamp': self.timestamp,
        }


class OrchestratorAgent:
    """
    Main orchestrator that routes tickets based on classification and RAG confidence.
    
    Workflow:
    1. Classify ticket (category, severity, keywords)
    2. Perform RAG retrieval with confidence scoring
    3. Decide routing path based on confidence
    4. Return analysis with routing recommendation
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize orchestrator.
        
        Args:
            vector_store: Chroma vector store (optional, initialized if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.rag_retriever = RAGRetriever(self.vector_store)
        
        # Configuration
        self.high_confidence_threshold = 0.85
        self.medium_confidence_threshold = 0.50
        self.low_confidence_threshold = 0.0
        
        LOGGER.info("OrchestratorAgent initialized")
    
    def classify_ticket(self, content: str) -> ClassificationResult:
        """
        Classify incoming ticket by category, severity, and keywords.
        
        Strategy:
        1. Search for category keywords
        2. Determine severity from language intensity
        3. Extract key terms
        4. Calculate classification confidence
        
        Args:
            content: Ticket text
            
        Returns:
            ClassificationResult with category, severity, keywords, confidence
        """
        content_lower = content.lower()
        
        # Category keywords mapping
        category_keywords = {
            TicketCategory.TECHNICAL: [
                'error', 'bug', 'crash', 'broken', 'not working', 'failed',
                'exception', 'issue', 'problem', 'doesn\'t work', 'api',
                'integration', 'code', 'technical', 'install', 'setup'
            ],
            TicketCategory.BILLING: [
                'charge', 'payment', 'invoice', 'refund', 'billing', 'credit',
                'money', 'cost', 'price', 'fee', 'subscription', 'cancel',
                'upgrade', 'downgrade', 'account', 'charged'
            ],
            TicketCategory.ACCOUNT: [
                'password', 'login', 'sign in', 'sign up', 'account',
                'username', 'email', 'verify', 'confirm', 'reset',
                'access', 'profile', 'settings', 'security'
            ],
            TicketCategory.FEATURE: [
                'feature', 'request', 'wish', 'like to have', 'would be nice',
                'enhancement', 'improvement', 'add', 'suggestion', 'idea'
            ],
            TicketCategory.BUG: [
                'bug', 'broken', 'error', 'crash', 'issue', 'problem',
                'unexpected', 'wrong', 'incorrect', 'malfunction'
            ],
        }
        
        # Severity keywords mapping
        severity_keywords = {
            TicketSeverity.CRITICAL: [
                'critical', 'urgent', 'emergency', 'production down',
                'cannot work', 'complete failure', 'blocked'
            ],
            TicketSeverity.HIGH: [
                'severe', 'major', 'important', 'urgent', 'asap',
                'broken', 'cannot', 'unable to'
            ],
            TicketSeverity.MEDIUM: [
                'issue', 'problem', 'doesn\'t work', 'need help',
                'please help'
            ],
            TicketSeverity.LOW: [
                'question', 'how do i', 'how to', 'information',
                'general inquiry'
            ]
        }
        
        # Classify category
        category_scores = {}
        for cat, keywords in category_keywords.items():
            match_count = sum(1 for kw in keywords if kw in content_lower)
            if match_count > 0:
                category_scores[cat] = match_count / len(keywords)
        
        if category_scores:
            category = max(category_scores, key=category_scores.get)
            category_confidence = category_scores[category]
        else:
            category = TicketCategory.UNKNOWN
            category_confidence = 0.3
        
        # Classify severity
        severity_scores = {}
        for sev, keywords in severity_keywords.items():
            match_count = sum(1 for kw in keywords if kw in content_lower)
            if match_count > 0:
                severity_scores[sev] = match_count / len(keywords)
        
        if severity_scores:
            severity = max(severity_scores, key=severity_scores.get)
        else:
            severity = TicketSeverity.MEDIUM
        
        # Extract keywords (words >3 chars that appear in category keywords)
        extracted_keywords = []
        words = content_lower.split()
        for cat_keywords in category_keywords.values():
            for kw in cat_keywords:
                if kw in content_lower and kw not in extracted_keywords:
                    extracted_keywords.append(kw)
        
        extracted_keywords = extracted_keywords[:10]  # Top 10
        
        overall_confidence = min(0.95, category_confidence + 0.1)
        
        result = ClassificationResult(
            category=category,
            severity=severity,
            keywords=extracted_keywords,
            confidence=overall_confidence
        )
        
        LOGGER.info(f"Classified ticket: {result.category.value} ({severity.value})")
        return result
    
    def decide_routing(
        self,
        classification: ClassificationResult,
        rag_context: RetrievalContext
    ) -> RoutingDecision:
        """
        Decide routing path based on classification and RAG confidence.
        
        Logic:
        - Confidence > 0.85: AUTO_RESPOND (simple FAQ)
        - Confidence 0.50-0.85: SINGLE_SPECIALIST (one agent can handle)
        - Confidence < 0.50: SPECIALIST_TEAM (escalate to team)
        
        Args:
            classification: Ticket classification
            rag_context: RAG retrieval context with confidence
            
        Returns:
            RoutingDecision with path and reasoning
        """
        confidence = rag_context.confidence_score
        
        # Determine routing path
        if confidence > self.high_confidence_threshold:
            path = RoutingPath.AUTO_RESPOND
            reason = f"High confidence ({confidence:.1%}) - FAQ-like question"
            specialist = None
            
        elif confidence >= self.medium_confidence_threshold:
            path = RoutingPath.SINGLE_SPECIALIST
            
            # Pick specialist based on category
            if classification.category == TicketCategory.BILLING:
                specialist = "billing_agent"
            elif classification.category == TicketCategory.TECHNICAL:
                specialist = "technical_agent"
            elif classification.category == TicketCategory.ACCOUNT:
                specialist = "technical_agent"  # Account issues to technical
            else:
                specialist = "technical_agent"  # Default to technical
            
            reason = f"Medium confidence ({confidence:.1%}) - Route to {specialist}"
            
        else:
            path = RoutingPath.SPECIALIST_TEAM
            specialist = None
            escalation_reason = (
                f"Low confidence ({confidence:.1%}) - Complex issue requires "
                "collaboration between Technical, Billing, and Resolution Drafter agents"
            )
            reason = escalation_reason
        
        decision = RoutingDecision(
            path=path,
            confidence_score=confidence,
            recommended_specialist=specialist,
            reason=reason,
            escalation_reason=escalation_reason if path == RoutingPath.SPECIALIST_TEAM else None
        )
        
        LOGGER.info(f"Routing decision: {path.value} (confidence: {confidence:.1%})")
        return decision
    
    def analyze_ticket(self, ticket_content: str, ticket_id: str = "TICKET-001") -> TicketAnalysis:
        """
        Perform complete ticket analysis and routing.
        
        Workflow:
        1. Classify ticket (category, severity, keywords)
        2. Perform RAG retrieval for context and confidence
        3. Decide routing path
        4. Return complete analysis
        
        Args:
            ticket_content: The ticket text
            ticket_id: Unique ticket identifier (default: TICKET-001)
            
        Returns:
            TicketAnalysis with classification, RAG context, and routing decision
        """
        LOGGER.info(f"Analyzing ticket {ticket_id}")
        
        # Step 1: Classify ticket
        classification = self.classify_ticket(ticket_content)
        
        # Step 2: Retrieve context using RAG
        rag_context = self.rag_retriever.retrieve_multi_step(
            ticket_content,
            k=RAG_K_RETRIEVAL
        )
        
        # Step 3: Decide routing
        routing_decision = self.decide_routing(classification, rag_context)
        
        # Combine results
        analysis = TicketAnalysis(
            ticket_id=ticket_id,
            content=ticket_content,
            classification=classification,
            rag_context=rag_context,
            routing_decision=routing_decision
        )
        
        LOGGER.info(f"Ticket analysis complete: {analysis.routing_decision.path.value}")
        return analysis
    
    def batch_analyze(
        self,
        tickets: List[Dict[str, str]]
    ) -> List[TicketAnalysis]:
        """
        Analyze multiple tickets.
        
        Args:
            tickets: List of dicts with 'id' and 'content' keys
            
        Returns:
            List of TicketAnalysis objects
        """
        results = []
        for ticket in tickets:
            analysis = self.analyze_ticket(
                ticket['content'],
                ticket.get('id', f"TICKET-{len(results)+1}")
            )
            results.append(analysis)
        
        LOGGER.info(f"Batch analysis complete: {len(results)} tickets")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            'high_confidence_threshold': self.high_confidence_threshold,
            'medium_confidence_threshold': self.medium_confidence_threshold,
            'low_confidence_threshold': self.low_confidence_threshold,
            'rag_retriever': self.rag_retriever.get_stats(),
        }
    
    def print_stats(self):
        """Print orchestrator statistics"""
        stats = self.get_stats()
        print("\n=== Orchestrator Stats ===")
        print(f"High confidence threshold: >{stats['high_confidence_threshold']}")
        print(f"Medium confidence threshold: {stats['medium_confidence_threshold']}-{stats['high_confidence_threshold']}")
        print(f"Low confidence threshold: <{stats['medium_confidence_threshold']}")
        print(f"\nRAG Retriever Stats:")
        print(f"  Total queries processed: {stats['rag_retriever'].get('total_queries', 0)}")
        print(f"  Average confidence: {stats['rag_retriever'].get('avg_confidence', 0):.2%}")


# Example usage
if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent()
    
    # Example ticket
    example_ticket = "I keep getting an error when trying to log in. Error code: 403"
    
    # Analyze
    analysis = orchestrator.analyze_ticket(example_ticket, "TICKET-001")
    
    print("\n=== Ticket Analysis ===")
    print(json.dumps(analysis.to_dict(), indent=2))
