# Quick Start Guide 🚀

## Phase 1 Setup Complete! ✅

Your project structure is ready. Follow these steps to get the bot running.

---

## Step 1: Install Ollama (One-time setup)

Ollama is the local LLM engine that runs completely offline.

### Download Ollama

Visit [ollama.ai](https://ollama.ai/) and download for your OS:

- **Windows**: Download `.exe` installer
- **macOS**: Download `.dmg` installer
- **Linux**: Follow Linux installation instructions

### Pull a Model

Open terminal/command prompt and run:

```bash
# Option 1: Mistral 7B (Recommended - Best quality, ~4GB)
ollama pull mistral

# Option 2: Llama2 7B (More conservative, ~4GB)
ollama pull llama2

# Option 3: Neural-chat (Fast, ~4GB)
ollama pull neural-chat
```

This downloads the model (~4GB) to your local machine. **Only needs to be done once.**

### Start Ollama Server

Keep this running in a terminal:

```bash
ollama serve
```

You should see:

```
Starting Ollama in server mode
Listening on http://localhost:11434
```

---

## Step 2: Setup Python Environment

### Windows

```cmd
cd "d:\Projects\AI Project\Intelligent FAQ & Ticket Resolution Bot"
setup.bat
```

This will:

- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create .env file
- ✅ Create necessary directories

### Linux/macOS

```bash
cd path/to/support-bot
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

---

## Step 3: Prepare Your Data

### Historical Tickets (CSV)

Create or place your CSV file in `data/tickets.csv`

### Product Documentation

Place docs in `data/docs/` directory

**Example structure:**

```
data/docs/
├── user_guide.pdf
├── faq.md
├── api_docs.txt
└── troubleshooting.docx
```

---

## Step 4: Configure (Optional)

Edit `.env` to customize settings:

```env
# Which model to use
OLLAMA_MODEL=mistral

# How creative responses should be (0-1, lower = more factual)
LLM_TEMPERATURE=0.3

# Maximum response length
LLM_MAX_TOKENS=2048
```

See `config.py` for all available options.

---

## Step 5: Run the Application

### Terminal 1: Keep Ollama Running

```bash
ollama serve
```

### Terminal 2: Run the Bot

**Windows:**

```cmd
cd "d:\Projects\AI Project\Intelligent FAQ & Ticket Resolution Bot"
venv\Scripts\activate.bat
streamlit run ui/streamlit_app.py
```

**Linux/macOS:**

```bash
cd path/to/support-bot
source venv/bin/activate
streamlit run ui/streamlit_app.py
```

### Expected Output:

```
  You can now view your Streamlit app in your browser.

  URL: http://localhost:8501

  Network URL: http://YOUR_IP:8501
```

Open your browser and visit **http://localhost:8501**

---

## Step 6: Test the Bot

### Using the Streamlit UI:

1. **Submit a Ticket**
   - Enter customer name
   - Select ticket category
   - Enter subject and description
   - Click "Submit Ticket"

2. **View Results**
   - See confidence score
   - Read generated response
   - View referenced documents
   - Check which agents processed it

### Test with Sample Data:

Try these sample queries:

**Simple FAQ-like:**

- "How do I enable two-factor authentication?"
- "I forgot my password, what do I do?"

**Billing:**

- "I was charged twice for my subscription"
- "Can I get a refund?"

**Technical:**

- "My API is getting timeout errors"
- "SSL certificate warning in browser"

---

## Architecture Overview (What Happens Behind the Scenes)

```
1. You submit a ticket
   ↓
2. Embeddings generated (sentence-transformers)
   ↓
3. RAG retrieves relevant docs (Chroma vector store)
   ↓
4. Orchestrator decides: Simple or Complex?
   ↓
   ├─ Simple (high confidence)
   │  └─→ Generate direct response (Ollama LLM)
   │
   └─ Complex (low confidence)
      └─→ Route to specialist agents
         ├─ Technical Agent
         ├─ Billing Agent
         └─ Resolution Drafter
            └─→ Synthesize final response
   ↓
5. Response displayed in UI
```

---

## Troubleshooting ⚠️

### ❌ "Connection refused: http://localhost:11434"

**Fix:** Make sure Ollama is running:

```bash
ollama serve
```

### ❌ "ModuleNotFoundError: No module named 'streamlit'"

**Fix:** Activate virtual environment:

```bash
# Windows
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

Then run dependencies install:

```bash
pip install -r requirements.txt
```

### ❌ "CUDA out of memory" or Slow responses

**Fix:** Use a smaller model:

```bash
ollama pull neural-chat  # Smaller, faster
```

Or reduce `LLM_MAX_TOKENS` in `.env`

### ❌ No results returned

**Fix:** Check your data:

1. Verify `data/tickets.csv` exists and has content
2. Verify `data/docs/` has documentation files
3. Check console for error messages

---

## System Requirements

- **CPU**: Modern processor (Intel i5+, AMD Ryzen 5+)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: 5GB free (for LLM) + space for data
- **Network**: Optional (fully offline after setup)

---

## Verify Everything Works

**Checklist:**

- [ ] Ollama installed and running
- [ ] Python virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created (can use defaults)
- [ ] Sample data in place
- [ ] Streamlit app runs without errors
- [ ] Can submit a test ticket
- [ ] Getting responses from bot

---

## Getting Help

Check the main **README.md** for:

- Detailed architecture
- API reference
- Configuration options
- Troubleshooting guide
- Development setup

---

**Ready to build? Let's go! 🚀**

If you hit issues, check troubleshooting section above or create an issue in the repository.
