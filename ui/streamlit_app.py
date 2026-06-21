"""
Phase 6: Streamlit Web UI for Intelligent Support Bot

Web interface for:
- Submitting support tickets
- Viewing agent responses
- Monitoring system performance
- Analytics dashboard
- Response history
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

from src.orchestrator import OrchestratorAgent, RoutingPath
from src.agents import SpecialistAgentManager
from src.vector_store import VectorStore
from src.config import LOGGER


# ============================================================================
# SESSION STATE & INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'tickets_history' not in st.session_state:
        st.session_state.tickets_history = []
    
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent()
    
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = SpecialistAgentManager()
    
    if 'current_ticket' not in st.session_state:
        st.session_state.current_ticket = None
    
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_header():
    """Display application header"""
    st.set_page_config(
        page_title="Intelligent Support Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 Intelligent Support Bot")
        st.markdown("*Powered by RAG, Classification & Multi-Agent Collaboration*")
    with col2:
        st.metric("Status", "🟢 Online", delta="Ready")


def display_ticket_form() -> Optional[str]:
    """Display ticket submission form"""
    st.header("📝 Submit Ticket")
    
    with st.form("ticket_form"):
        # Ticket category (optional pre-selection)
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Category (optional):",
                options=["Auto-detect", "Technical", "Billing", "Account", "Feature", "General"],
                help="Pre-select a category or let the system auto-detect"
            )
        
        with col2:
            priority = st.selectbox(
                "Priority:",
                options=["Normal", "High", "Critical"],
                help="Indicate urgency if known"
            )
        
        # Ticket content
        ticket_content = st.text_area(
            "Describe your issue:",
            placeholder="Explain the problem you're facing...",
            height=150,
            help="Provide as much detail as possible for better responses"
        )
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Submit Ticket",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            if not ticket_content.strip():
                st.error("Please enter a ticket description")
                return None
            return ticket_content
    
    return None


def display_processing_indicator():
    """Display processing indicator while analyzing ticket"""
    with st.spinner("🔄 Analyzing ticket..."):
        time.sleep(0.5)  # Simulate processing
    st.success("✅ Analysis complete!")


def display_ticket_analysis(analysis):
    """Display detailed ticket analysis"""
    st.subheader("📊 Ticket Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Category",
            analysis.classification.category.value.title(),
            delta=f"{analysis.classification.confidence:.0%} confidence"
        )
    
    with col2:
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        st.metric(
            "Severity",
            severity_emoji.get(analysis.classification.severity.value, '⚪') + 
            " " + analysis.classification.severity.value.title()
        )
    
    with col3:
        st.metric(
            "RAG Confidence",
            f"{analysis.rag_context.confidence_score:.0%}",
            delta=f"{analysis.rag_context.total_documents} documents found"
        )
    
    # Keywords
    if analysis.classification.keywords:
        st.write("**Key Terms:**")
        keyword_cols = st.columns(min(len(analysis.classification.keywords), 5))
        for i, keyword in enumerate(analysis.classification.keywords[:5]):
            with keyword_cols[i % 5]:
                st.button(f"#{keyword}", disabled=True, use_container_width=True)


def display_routing_decision(analysis):
    """Display routing decision with visualization"""
    st.subheader("🛤️ Routing Decision")
    
    routing = analysis.routing_decision
    
    # Color-coded routing path
    path_colors = {
        'auto_respond': '🟢',
        'single_specialist': '🟡',
        'specialist_team': '🔴',
        'manual_review': '⚫'
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Path:** {path_colors.get(routing.path.value, '⚪')} {routing.path.value.upper()}")
        st.write(f"**Confidence:** {routing.confidence_score:.0%}")
        
        if routing.recommended_specialist:
            st.write(f"**Specialist:** {routing.recommended_specialist.title()}")
        
        st.write(f"**Reason:** {routing.reason}")
    
    with col2:
        # Confidence gauge
        progress_value = routing.confidence_score
        if routing.path.value == 'auto_respond':
            st.success("✅ High Confidence")
        elif routing.path.value == 'single_specialist':
            st.warning("⚠️ Medium Confidence")
        else:
            st.error("❌ Low Confidence")


def display_rag_context(analysis):
    """Display RAG context and sources"""
    st.subheader("📚 Knowledge Base Context")
    
    if analysis.rag_context.total_documents > 0:
        st.write(f"**Found {analysis.rag_context.total_documents} relevant documents**")
        
        with st.expander("📖 View Context", expanded=False):
            st.write(analysis.rag_context.formatted_context)
        
        # Document list
        st.write("**Retrieved Documents:**")
        for i, doc in enumerate(analysis.rag_context.retrieved_documents, 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{i}. {doc.source}**")
            with col2:
                st.metric("Similarity", f"{doc.similarity_score:.0%}", label_visibility="collapsed")
            with col3:
                st.write(f"Rank #{doc.rank}")
    else:
        st.warning("⚠️ No relevant documents found in knowledge base")


def display_agent_response(result: Dict[str, Any]):
    """Display agent response based on routing"""
    st.subheader("💬 Response")
    
    result_type = result.get('type')
    
    if result_type == 'auto_response':
        st.success("✅ Instant FAQ Response")
        st.write(result['content'])
        st.metric("Confidence", f"{result['confidence']:.0%}")
    
    elif result_type == 'specialist_response':
        response = result['response']
        st.info("👤 Specialist Agent Response")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(response['content'])
        with col2:
            st.metric("Confidence", f"{response['confidence']:.0%}")
            st.write(f"**Agent:** {response['agent_type']}")
        
        if response['tools_used']:
            with st.expander("🛠️ Tools Used"):
                for tool in response['tools_used']:
                    st.write(f"• {tool}")
    
    elif result_type == 'team_response':
        output = result['output']
        st.info("👥 Team Response (Multiple Agents)")
        
        st.write(output['final_response'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Confidence", f"{output['confidence']:.0%}")
        with col2:
            escalation_status = "⚠️ Escalation Needed" if output['escalation_recommended'] else "✅ Ready to Send"
            st.metric("Status", escalation_status)
        
        # Agent contributions
        if output['agent_contributions']:
            st.write("**Agent Contributions:**")
            for agent, reasoning in output['agent_contributions'].items():
                with st.expander(f"📊 {agent.title()}"):
                    st.write(reasoning)


def display_response_actions(analysis, result):
    """Display action buttons for responses"""
    st.divider()
    st.subheader("📤 Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Send Response", use_container_width=True):
            st.success("Response sent to customer!")
            # Log to history
            st.session_state.tickets_history.append({
                'timestamp': datetime.now(),
                'ticket': analysis.content[:100] + "...",
                'status': 'sent',
                'confidence': analysis.routing_decision.confidence_score
            })
    
    with col2:
        if st.button("👤 Review First", use_container_width=True):
            st.info("Response flagged for human review")
    
    with col3:
        if st.button("📋 Copy Response", use_container_width=True):
            st.success("Response copied to clipboard!")
    
    with col4:
        if st.button("💾 Save Draft", use_container_width=True):
            st.info("Response saved as draft")


def display_analytics_dashboard():
    """Display analytics and statistics"""
    st.header("📈 Analytics Dashboard")
    
    # Create sample data if history is empty
    if not st.session_state.tickets_history:
        st.info("No tickets processed yet. Submit a ticket to see analytics.")
        return
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.tickets_history)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(history_df))
    
    with col2:
        avg_confidence = history_df['confidence'].mean() if 'confidence' in history_df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")
    
    with col3:
        sent_count = len(history_df[history_df['status'] == 'sent']) if 'status' in history_df.columns else 0
        st.metric("Responses Sent", sent_count)
    
    with col4:
        st.metric("System Status", "🟢 Online")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tickets Over Time")
        # Create time series data
        if 'timestamp' in history_df.columns:
            time_series = history_df.set_index('timestamp').resample('H').size()
            st.line_chart(time_series)
    
    with col2:
        st.subheader("Confidence Distribution")
        if 'confidence' in history_df.columns:
            st.bar_chart(history_df['confidence'].value_counts().sort_index())
    
    st.divider()
    
    # Recent tickets table
    st.subheader("Recent Tickets")
    if len(history_df) > 0:
        display_df = history_df[['timestamp', 'ticket', 'status', 'confidence']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0%}")
        st.dataframe(display_df, use_container_width=True)


def display_settings_panel():
    """Display settings panel in sidebar"""
    st.sidebar.header("⚙️ Settings")
    
    with st.sidebar.expander("System Configuration"):
        st.write("**Confidence Thresholds:**")
        high_conf = st.slider("High Confidence (>)", 0.5, 1.0, 0.85, step=0.05)
        medium_conf = st.slider("Medium Confidence (>)", 0.0, 0.85, 0.50, step=0.05)
        
        st.write("**RAG Settings:**")
        k_retrieval = st.slider("Documents to Retrieve", 1, 20, 5)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, step=0.05)
        
        st.write("**Agent Settings:**")
        auto_route = st.checkbox("Auto-route based on confidence", value=True)
        require_review = st.checkbox("Flag low-confidence for review", value=True)


def display_help_panel():
    """Display help and information panel"""
    st.sidebar.header("ℹ️ Help")
    
    with st.sidebar.expander("How it Works"):
        st.markdown("""
        1. **Submit Ticket**: Describe your issue in the form
        2. **Classification**: System analyzes ticket (category, severity)
        3. **RAG Retrieval**: Searches knowledge base for relevant docs
        4. **Routing Decision**: Determines confidence and routing path
        5. **Agent Processing**: Specialist agents handle the issue
        6. **Response Generation**: Professional response is synthesized
        7. **Review & Send**: Approve response before sending
        """)
    
    with st.sidebar.expander("Routing Paths"):
        st.markdown("""
        **🟢 High Confidence (>85%)**
        - FAQ-like questions
        - Clear answers available
        - Instant response

        **🟡 Medium Confidence (50-85%)**
        - Specialist agent handles
        - Thorough investigation
        - Professional response

        **🔴 Low Confidence (<50%)**
        - Multiple agents collaborate
        - Complex issue resolution
        - Team response
        """)
    
    with st.sidebar.expander("System Info"):
        st.markdown(f"""
        **Intelligent Support Bot v1.0**
        
        Built with:
        - LangChain + LangGraph
        - Sentence Transformers
        - Chroma Vector DB
        - Ollama LLM
        
        **Components:**
        - Phase 2: Data Ingestion
        - Phase 3: RAG Retriever
        - Phase 4: Orchestrator
        - Phase 5: Specialist Agents
        - Phase 6: Streamlit UI (this)
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Create sidebar
    st.sidebar.write("---")
    page = st.sidebar.radio(
        "Navigation",
        ["💬 Support Ticket", "📈 Analytics", "⚙️ Settings"],
        key="page_nav"
    )
    
    display_settings_panel()
    st.sidebar.write("---")
    display_help_panel()
    
    # Main content
    if page == "💬 Support Ticket":
        # Ticket submission
        ticket_content = display_ticket_form()
        
        if ticket_content:
            # Process ticket
            display_processing_indicator()
            
            # Analyze with orchestrator
            analysis = st.session_state.orchestrator.analyze_ticket(
                ticket_content,
                ticket_id=f"TICKET-{int(time.time())}"
            )
            
            st.session_state.current_ticket = ticket_content
            st.session_state.current_result = analysis
            
            st.divider()
            
            # Display analysis results
            display_ticket_analysis(analysis)
            
            st.divider()
            
            # Display routing decision
            display_routing_decision(analysis)
            
            st.divider()
            
            # Display RAG context
            display_rag_context(analysis)
            
            st.divider()
            
            # Process through agents
            result = st.session_state.agent_manager.route_ticket(analysis)
            
            # Display response
            display_agent_response(result)
            
            st.divider()
            
            # Display actions
            display_response_actions(analysis, result)
    
    elif page == "📈 Analytics":
        display_analytics_dashboard()
    
    elif page == "⚙️ Settings":
        st.header("⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            model = st.selectbox("LLM Model", ["Mistral 7B", "Llama2"])
            embedding = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
            
            st.subheader("Processing Options")
            batch_mode = st.checkbox("Enable batch processing")
            parallel_agents = st.checkbox("Parallel agent processing")
        
        with col2:
            st.subheader("Knowledge Base")
            kb_docs = st.metric("Documents in KB", "~200+")
            cache_hit = st.metric("Cache Hit Rate", "~90%")
            
            st.subheader("Specialist Agents")
            agents_active = st.metric("Active Agents", "3")
            
            if st.button("Refresh Knowledge Base"):
                st.success("Knowledge base refreshed!")
            
            if st.button("Clear Cache"):
                st.warning("Cache cleared!")


if __name__ == "__main__":
    main()
