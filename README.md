# Offline Support Bot 🤖

An intelligent offline FAQ & Ticket Resolution Bot that processes historical support tickets and product documentation to automatically classify and resolve customer issues. Complex tickets are escalated to specialized agent teams (Technical, Billing, Resolution Drafter) for collaborative problem-solving.

## Features ✨

- **Fully Offline**: All processing runs locally—no cloud APIs or internet required
- **RAG-Powered**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Multi-Agent Architecture**: Specialized agents for technical, billing, and resolution drafting
- **Automatic Classification**: Identifies ticket category and complexity automatically
- **Smart Escalation**: Routes complex issues to agent teams for collaboration
- **Web Interface**: Streamlit UI for easy ticket submission and monitoring
- **CSV & Document Support**: Ingests historical tickets and product documentation (PDF, DOCX, TXT, MD)
- **Free & Open Source**: Built with Ollama, LangChain, Chroma, and other free tools

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                          │
│          (Ticket Input, Results Display, Dashboard)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR AGENT                           │
│     (Classification, Confidence Scoring, Routing)            │
└──────┬──────────────────────────┬──────────────────────────┘
       │                          │
       ├─ High Confidence ─────────┤─ Low Confidence ───────────┐
       ▼                           ▼                             ▼
    ┌──────────┐        ┌──────────────────────┐     ┌────────────────────┐
    │ Direct   │        │  SPECIALIST AGENTS   │     │  SPECIALIST AGENTS │
    │ Response │        │  (Parallel/Serial)   │     │  (Parallel/Serial) │
    │ Generator│        │                      │     │                    │
    │          │        │ ┌──────────────────┐ │     │ ┌──────────────┐   │
    │          │        │ │Technical Agent   │ │     │ │Billing Agent │   │
    │          │        │ └──────────────────┘ │     │ └──────────────┘   │
    │          │        │ ┌──────────────────┐ │     │                    │
    │          │        │ │Resolution Drafter│ │     │                    │
    │          │        │ └──────────────────┘ │     │                    │
    │          │        └──────────────────────┘     └────────────────────┘
    │          │                   │
    │          │                   ▼
    │          │        ┌──────────────────────┐
    │          │        │ Response Synthesizer │
    │          │        └──────────────────────┘
    │          │                   │
    └──────────┴───────────────────┴────────────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ Vector Store (Chroma)        │
                   │ Knowledge Base from:          │
                   │ - Historical Tickets (CSV)   │
                   │ - Product Docs (PDF/DOCX)    │
                   └──────────────────────────────┘
                                  △
                                  │
                        ┌─────────────────────┐
                        │  Sentence Transformer│
                        │  (Embeddings)       │
                        └─────────────────────┘
                                  △
                                  │
                        ┌─────────────────────┐
                        │  Ollama (Local LLM) │
                        │  Mistral/Llama2     │
                        └─────────────────────┘
```

## Quick Start 🚀

### Prerequisites

- Python 3.9+
- Ollama installed ([Download](https://ollama.ai/))
- 8GB+ RAM recommended
- ~5GB disk space for Ollama models

### Step 1: Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/)

### Step 2: Pull a Local Model

```bash
# Pull Mistral (7B, ~4GB)
ollama pull mistral

# Or use other models:
# ollama pull llama2          # Llama2 (7B, ~4GB)
# ollama pull neural-chat     # Neural-chat (7B, ~4GB)
```

### Step 3: Start Ollama Server

```bash
ollama serve
```
Keep this running in the background.

### Step 4: Clone and Setup Project

**On Windows:**
```bash
setup.bat
```

**On Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 5: Prepare Data

Place your data files in the `data/` directory:

**Historical Tickets (CSV):**
```
data/tickets.csv
```

Required columns: `id, customer_id, subject, description, category, resolution, timestamp`

**Product Documentation:**
```
data/docs/
├── user_guide.pdf
├── api_docs.md
├── faq.txt
└── troubleshooting.docx
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### Step 6: Run the Application

```bash
# Activate virtual environment (if not already)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Start the Streamlit app
streamlit run ui/streamlit_app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
support-bot/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration & constants
│   ├── logger.py                 # Logging setup
│   ├── data_processor.py         # CSV & doc ingestion
│   ├── embeddings.py             # Embedding generation
│   ├── vector_store.py           # Chroma integration
│   ├── rag_retriever.py          # RAG query logic
│   ├── orchestrator.py           # Main routing agent
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── technical_agent.py    # Technical support agent
│   │   ├── billing_agent.py      # Billing agent
│   │   └── resolution_drafter.py # Response synthesis
│   └── tools/
│       ├── __init__.py
│       └── agent_tools.py        # Agent tool definitions
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py          # Web interface
├── data/
│   ├── tickets.csv               # Historical tickets
│   ├── docs/                     # Product documentation
│   ├── sample/                   # Sample data
│   └── knowledge_base/           # Vector store & embeddings
├── tests/
│   └── (test files)
├── requirements.txt              # Dependencies
├── config.py                     # Main config
├── setup.bat                     # Windows setup
├── setup.sh                      # Linux/macOS setup
└── README.md                     # This file
```

## Configuration 🔧

Edit `.env` to customize settings:

```env
# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral              # Model to use
OLLAMA_TIMEOUT=120

# LLM generation settings
LLM_TEMPERATURE=0.3               # Lower = more deterministic
LLM_MAX_TOKENS=2048
LLM_TOP_P=0.9

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Free, lightweight

# Logging
LOG_LEVEL=INFO
```

## How It Works 🔄

### 1. **Data Ingestion**
   - Loads historical tickets from CSV
   - Parses product documentation (PDF, DOCX, TXT, MD)
   - Chunks documents (512 tokens, 50% overlap)
   - Generates embeddings using Sentence Transformers

### 2. **Knowledge Base**
   - Stores embeddings in local Chroma vector database
   - Enables semantic search over documents & tickets
   - No external APIs or cloud storage required

### 3. **Ticket Processing**
   - User submits new support ticket
   - Orchestrator retrieves relevant context via RAG
   - Calculates confidence score based on retrieval

### 4. **Decision Logic**
   - **High Confidence (> 0.8)**: Generate direct response
   - **Low Confidence (< 0.8)**: Escalate to specialist agents

### 5. **Specialist Agents**
   - **Technical Agent**: Handles technical troubleshooting
   - **Billing Agent**: Handles pricing, refunds, subscriptions
   - **Resolution Drafter**: Synthesizes multi-agent responses
   - Agents collaborate, share context, generate final response

### 6. **Response Delivery**
   - Final response shown in Streamlit UI
   - Confidence score and sources displayed
   - Response stored for future reference

## API Reference 📚

### Key Modules

#### `src.data_processor`
```python
from src.data_processor import DataProcessor

processor = DataProcessor()
tickets = processor.load_tickets("data/tickets.csv")
docs = processor.load_documentation("data/docs")
chunks = processor.chunk_documents(docs)
```

#### `src.embeddings`
```python
from src.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()
embedding = generator.embed_text("your text here")
embeddings = generator.embed_batch(list_of_texts)
```

#### `src.vector_store`
```python
from src.vector_store import VectorStore

store = VectorStore()
store.add_documents(chunks_with_embeddings)
results = store.search_by_text("search query", k=5)
```

#### `src.rag_retriever`
```python
from src.rag_retriever import RAGRetriever

retriever = RAGRetriever()
context = retriever.retrieve("customer question", k=5)
```

#### `src.orchestrator`
```python
from src.orchestrator import OrchestratorAgent

agent = OrchestratorAgent()
response = agent.process_ticket(
    subject="Issue with payment",
    description="Cannot process payment..."
)
```

## Development 💻

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ ui/

# Lint code
flake8 src/ ui/
```

### Adding Custom Agents

Create a new agent in `src/agents/`:

```python
# src/agents/custom_agent.py
from langchain.agents import Tool
from langchain import LLMChain
from src.config import OLLAMA_HOST, OLLAMA_MODEL_NAME

class CustomAgent:
    def __init__(self):
        self.llm = ...  # Initialize LLM
        self.tools = self._create_tools()
        
    def _create_tools(self):
        return [
            Tool(name="...", func=..., description="...")
        ]
    
    def run(self, query: str) -> str:
        # Agent logic
        pass
```

## Troubleshooting 🔧

### Ollama Connection Error
```
Error: Failed to connect to Ollama at http://localhost:11434
```
**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### Out of Memory Error
**Solution**: Use a smaller model:
```bash
ollama pull neural-chat  # 7B model, ~4GB
```

Or increase system swap space.

### Slow Embeddings
**Solution**: The first embedding generation is slow (downloads model). Subsequent runs are cached.

### Empty Vector Store
**Solution**: Ensure your data files are in the correct format:
- CSV must have required columns
- Documents must be in supported formats (.pdf, .docx, .txt, .md)

## Performance Tips ⚡

1. **Use GPU**: Enable GPU in Ollama for faster LLM inference
2. **Cache Embeddings**: Embeddings are cached after first generation
3. **Batch Processing**: Process multiple tickets in parallel
4. **Document Chunking**: Adjust `CHUNK_SIZE` in config based on your needs

## Limitations ⚠️

- **First Run**: Downloading embeddings model (~50MB) and LLM (~4-7GB) takes time
- **Response Time**: Typical response: 5-15 seconds (depends on LLM size & hardware)
- **Context Window**: Limited by LLM context window (varies by model)
- **No Web Search**: Uses only local knowledge base

## Future Enhancements 🎯

- [ ] Fine-tune embeddings on domain-specific data
- [ ] Multi-language support
- [ ] Streaming responses
- [ ] Feedback loop for model improvement
- [ ] Advanced analytics & reporting
- [ ] API endpoint for integration
- [ ] Docker containerization
- [ ] Database backend (PostgreSQL + pgvector)

## License 📄

MIT License - feel free to use this for personal or commercial projects.

## Support & Contributing 🤝

Found a bug? Have a feature request? Create an issue or submit a PR!

## Resources 📖

- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

**Built with ❤️ using free, open-source tools**
