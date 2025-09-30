# Task 2: RAG + LLM System for Technical Documentation

## 📋 Overview
A Retrieval-Augmented Generation (RAG) system that enables natural language querying of technical documentation (PDFs) using open-source models.

**Key Features:**
- ✅ PDF document ingestion and preprocessing
- ✅ Semantic search with FAISS vector store
- ✅ Citation-enforced answer generation
- ✅ Guardrails against hallucinations
- ✅ 100% open-source and free

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install required libraries
pip install sentence-transformers faiss-cpu PyPDF2
pip install transformers torch
pip install numpy pandas

# For GPU support (optional, faster):
pip install faiss-gpu
```

### Running the Prototype
```bash
# 1. Add your PDF files to the docs/ folder
# 2. Run the prototype
python rag_prototype.py

# The system will:
# - Ingest and index all PDFs (first run only)
# - Run example queries
# - Start interactive Q&A mode
```

---

## 📂 Folder Structure

```
Task2/
├── rag_prototype.py              # Main system implementation
├── README.md                     # This file
├── notes.md                      # Design document
├── architecture_diagram.pptx     # System architecture visual
├── docs/                         # Input PDFs (add your files here)
│   ├── manual_1.pdf
│   ├── sop_2.pdf
│   └── ...
├── rag_index/                    # Generated index (auto-created)
│   ├── faiss.index
│   └── chunks.pkl
└── evaluation.csv                # (Optional) Retrieval metrics
```

---

## 🏗️ System Architecture

### Components:

1. **Document Ingestion**
   - Extract text from PDFs using PyPDF2
   - Clean and normalize text
   - Split into overlapping chunks (512 tokens, 50 overlap)

2. **Embeddings & Indexing**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Vector Store: FAISS (IndexFlatL2)
   - Dimension: 384

3. **Retrieval Layer**
   - Dense vector search (semantic similarity)
   - Returns top-K most relevant chunks
   - Distance-based relevance threshold

4. **LLM Generation**
   - Model: `google/flan-t5-base` (or flan-t5-large)
   - Prompt engineering with retrieved context
   - Citation enforcement in prompt

5. **Guardrails**
   - Relevance threshold (distance > 1.5 → "Cannot answer")
   - Source citation enforcement
   - Confidence scoring (high/medium/low)

---

## 🎯 Design Decisions

### Chunking Strategy
**Choice**: 512 tokens with 50-token overlap

**Rationale**:
- 512 tokens captures complete context (paragraphs/sections)
- 50-token overlap prevents information loss at boundaries
- Balances retrieval granularity vs. computational cost

**Alternative**: Semantic chunking by paragraphs (more complex, minimal gain)

### Embedding Model
**Choice**: `all-MiniLM-L6-v2`

**Rationale**:
- Fast inference (50ms per query)
- Good quality (MTEB benchmark: 58.8)
- Small size (80MB) - easy deployment
- Free and open-source

**Alternatives**:
- `all-mpnet-base-v2`: Higher quality (63.3 MTEB) but slower
- `instructor-large`: Task-specific but requires more compute

### Retrieval Method
**Choice**: Dense vector search (FAISS)

**Rationale**:
- Handles semantic similarity well
- Fast search (< 10ms for 10K documents)
- Easy to implement and maintain

**Enhancement (Future)**: Hybrid search (BM25 + Dense)
- Combines keyword matching with semantic search
- Improves recall for exact terminology

### LLM Selection
**Choice**: Flan-T5-Base (250M parameters)

**Rationale**:
- Instruction-tuned for Q&A tasks
- Runs on CPU (no GPU required)
- Good summarization ability
- Free via Hugging Face

**Alternatives**:
- Flan-T5-Large: Better quality, needs GPU
- Llama-2-7B: Best quality but requires 16GB+ RAM
- GPT4All: Fully local but slower

---

##