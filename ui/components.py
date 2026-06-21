"""
UI Components and Utilities for Streamlit Application

Reusable components for:
- Metrics display
- Formatting functions
- Color coding
- Data visualization helpers
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

class ConfidenceFormatter:
    """Format confidence scores with colors and emojis"""
    
    @staticmethod
    def format_confidence(score: float) -> str:
        """Format confidence with emoji indicator"""
        if score > 0.85:
            return f"🟢 {score:.0%} (High)"
        elif score > 0.50:
            return f"🟡 {score:.0%} (Medium)"
        else:
            return f"🔴 {score:.0%} (Low)"
    
    @staticmethod
    def get_confidence_color(score: float) -> str:
        """Get color for confidence score"""
        if score > 0.85:
            return "#00CC00"  # Green
        elif score > 0.50:
            return "#FFAA00"  # Orange
        else:
            return "#CC0000"  # Red


class CategoryFormatter:
    """Format ticket categories with emojis"""
    
    EMOJI_MAP = {
        'technical': '🛠️',
        'billing': '💰',
        'account': '👤',
        'feature': '✨',
        'bug': '🐛',
        'general': '❓',
        'unknown': '❓'
    }
    
    @staticmethod
    def format_category(category: str) -> str:
        """Format category with emoji"""
        emoji = CategoryFormatter.EMOJI_MAP.get(category.lower(), '❓')
        return f"{emoji} {category.title()}"


class SeverityFormatter:
    """Format severity levels with colors and emojis"""
    
    EMOJI_MAP = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    }
    
    @staticmethod
    def format_severity(severity: str) -> str:
        """Format severity with emoji"""
        emoji = SeverityFormatter.EMOJI_MAP.get(severity.lower(), '⚪')
        return f"{emoji} {severity.title()}"


class RoutingFormatter:
    """Format routing paths with descriptions"""
    
    DESCRIPTIONS = {
        'auto_respond': '✅ FAQ Response (Instant)',
        'single_specialist': '👤 Specialist Agent',
        'specialist_team': '👥 Team Collaboration',
        'manual_review': '👨‍💼 Manual Review'
    }
    
    @staticmethod
    def format_path(path: str) -> str:
        """Format routing path"""
        return RoutingFormatter.DESCRIPTIONS.get(path, path)


# ============================================================================
# DISPLAY COMPONENTS
# ============================================================================

def display_confidence_gauge(score: float, label: str = "Confidence") -> None:
    """Display confidence score as gauge"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric(label, f"{score:.0%}")
    
    with col2:
        st.progress(score)
    
    with col3:
        st.write(ConfidenceFormatter.format_confidence(score))


def display_ticket_summary(ticket_id: str, content: str, confidence: float) -> None:
    """Display ticket summary card"""
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{ticket_id}**")
            st.write(content[:100] + "..." if len(content) > 100 else content)
        
        with col2:
            st.metric("Confidence", f"{confidence:.0%}", label_visibility="collapsed")
        
        with col3:
            st.write(f"✅ Processed")


def display_agent_card(agent_type: str, response_content: str, confidence: float) -> None:
    """Display agent response card"""
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.subheader(f"👤 {agent_type.title()} Agent")
            st.write(response_content)
        
        with col2:
            st.metric("Confidence", f"{confidence:.0%}")


def display_source_document(source: str, content: str, similarity: float, rank: int) -> None:
    """Display a source document from RAG context"""
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**#{rank}. {source}**")
        
        with col2:
            st.metric("Relevance", f"{similarity:.0%}", label_visibility="collapsed")
        
        with col3:
            if st.button("📋", key=f"copy_{rank}"):
                st.success("Copied!")
        
        with st.expander("View content"):
            st.write(content[:300] + "..." if len(content) > 300 else content)


# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

def display_session_info() -> None:
    """Display session information in sidebar"""
    with st.sidebar:
        st.write("---")
        st.subheader("📊 Session Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tickets", "0")
        with col2:
            st.metric("Avg Confidence", "0%")
        
        st.write("---")


def display_quick_actions() -> Optional[str]:
    """Display quick action buttons"""
    st.sidebar.write("---")
    st.sidebar.subheader("⚡ Quick Actions")
    
    action = None
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🆕 New Ticket", use_container_width=True):
            action = "new_ticket"
    
    with col2:
        if st.button("📜 History", use_container_width=True):
            action = "history"
    
    return action


def display_system_status() -> None:
    """Display system status"""
    st.sidebar.write("---")
    st.sidebar.subheader("🔧 System Status")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Orchestrator", "🟢 Ready")
    
    with col2:
        st.metric("Agents", "🟢 Ready")


# ============================================================================
# DATA FORMATTING
# ============================================================================

def format_analysis_for_display(analysis) -> Dict[str, Any]:
    """Format analysis object for display"""
    return {
        'ticket_id': analysis.ticket_id,
        'category': CategoryFormatter.format_category(analysis.classification.category.value),
        'severity': SeverityFormatter.format_severity(analysis.classification.severity.value),
        'classification_confidence': f"{analysis.classification.confidence:.0%}",
        'rag_confidence': f"{analysis.rag_context.confidence_score:.0%}",
        'documents_found': analysis.rag_context.total_documents,
        'routing_path': RoutingFormatter.format_path(analysis.routing_decision.path.value),
        'routing_confidence': f"{analysis.routing_decision.confidence_score:.0%}",
        'recommended_specialist': analysis.routing_decision.recommended_specialist or "N/A",
    }


def format_response_for_display(result) -> Dict[str, Any]:
    """Format result for display"""
    result_type = result.get('type', 'unknown')
    
    if result_type == 'auto_response':
        return {
            'type': 'Auto-Response',
            'content': result.get('content', ''),
            'confidence': f"{result.get('confidence', 0):.0%}",
        }
    
    elif result_type == 'specialist_response':
        response = result.get('response', {})
        return {
            'type': 'Specialist Response',
            'agent': response.get('agent_type', 'Unknown'),
            'content': response.get('content', ''),
            'confidence': f"{response.get('confidence', 0):.0%}",
            'tools_used': response.get('tools_used', []),
        }
    
    elif result_type == 'team_response':
        output = result.get('output', {})
        return {
            'type': 'Team Response',
            'content': output.get('final_response', ''),
            'confidence': f"{output.get('confidence', 0):.0%}",
            'sources': output.get('sources', []),
            'escalation': output.get('escalation_recommended', False),
        }
    
    return {'type': 'Unknown', 'content': 'Unable to format response'}


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_json(analysis, result) -> str:
    """Export ticket and response to JSON"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'ticket_id': analysis.ticket_id,
        'ticket_content': analysis.content,
        'analysis': format_analysis_for_display(analysis),
        'response': format_response_for_display(result),
    }
    return json.dumps(export_data, indent=2)


def export_to_markdown(analysis, result) -> str:
    """Export ticket and response to Markdown"""
    formatted_analysis = format_analysis_for_display(analysis)
    formatted_response = format_response_for_display(result)
    
    md = f"""# Support Ticket #{analysis.ticket_id}

## Ticket Information
- **Submitted:** {datetime.now().isoformat()}
- **Category:** {formatted_analysis['category']}
- **Severity:** {formatted_analysis['severity']}

## Content
{analysis.content}

## Analysis
- **Classification Confidence:** {formatted_analysis['classification_confidence']}
- **RAG Confidence:** {formatted_analysis['rag_confidence']}
- **Documents Found:** {formatted_analysis['documents_found']}

## Routing Decision
- **Path:** {formatted_analysis['routing_path']}
- **Confidence:** {formatted_analysis['routing_confidence']}
- **Specialist:** {formatted_analysis['recommended_specialist']}

## Response
### {formatted_response['type']}
Confidence: {formatted_response['confidence']}

{formatted_response['content']}

---
*Generated by Intelligent Support Bot v1.0*
"""
    return md


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_ticket_input(content: str) -> tuple[bool, str]:
    """Validate ticket input"""
    if not content:
        return False, "Ticket cannot be empty"
    
    if len(content) < 10:
        return False, "Ticket must be at least 10 characters"
    
    if len(content) > 5000:
        return False, "Ticket cannot exceed 5000 characters"
    
    return True, "✅ Valid"


# ============================================================================
# CACHE & SESSION MANAGEMENT
# ============================================================================

def clear_cache() -> None:
    """Clear application cache"""
    st.cache_data.clear()
    st.cache_resource.clear()


def reset_session() -> None:
    """Reset session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
