# RAG System Architecture Diagram

## 🏗️ System Overview (ASCII Visualization)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG SYSTEM ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INPUT LAYER   │    │ PROCESSING LAYER│    │ RETRIEVAL LAYER │    │  OUTPUT LAYER   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

     ┌─────────┐
     │   PDF   │
     │ Documents│ ──┐
     └─────────┘   │
                   │    ┌─────────────────┐
     ┌─────────┐   │    │                 │    ┌─────────────────┐
     │   PDF   │ ──┼───▶│  PyPDF2 Text    │───▶│   Text Chunks   │
     │ Documents│   │    │   Extraction    │    │  (512 tokens)   │
     └─────────┘   │    │                 │    │  (50 overlap)   │
                   │    └─────────────────┘    └─────────────────┘
     ┌─────────┐   │                                    │
     │   PDF   │ ──┘                                    │
     │ Documents│                                       ▼
     └─────────┘            
                           ┌─────────────────┐    ┌─────────────────┐
     ┌─────────┐           │                 │    │                 │
     │  User   │ ─────────▶│ Query Encoding  │───▶│  Sentence-BERT  │
     │ Question│           │                 │    │   Embeddings    │
     └─────────┘           └─────────────────┘    │ (768 dimensions)│
                                                  └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            VECTOR SEARCH & RETRIEVAL                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │    FAISS    │    │  Semantic   │    │CrossEncoder │    │    Top-K    │    │
│  │Vector Index │───▶│   Search    │───▶│ Re-ranking  │───▶│  Relevant   │    │
│  │ (L2 Dist.)  │    │ (Top-10)    │    │(Top-3 Best) │    │   Chunks    │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ANSWER GENERATION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Context   │    │   Prompt    │    │  FLAN-T5    │    │   Final     │    │
│  │ Injection   │───▶│ Engineering │───▶│   Large     │───▶│   Answer    │    │
│  │& Metadata   │    │& Citations  │    │ (780M params)│    │+ Citations  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GUARDRAILS & QA                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  Relevance  │    │   Citation  │    │ Confidence  │    │   Output    │    │
│  │ Filtering   │───▶│ Enforcement │───▶│   Scoring   │───▶│ Validation  │    │
│  │(Score > 0.5)│    │& Source Map │    │ (H/M/L)     │    │& Safety     │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
                                          ┌─────────────────┐
                                          │  USER RECEIVES  │
                                          │                 │
                                          │ • Answer Text   │
                                          │ • Source Cites  │
                                          │ • Confidence    │
                                          │ • Debug Info    │
                                          └─────────────────┘
```

## 🔄 Data Flow Sequence

```
1. DOCUMENT INGESTION
   PDF Files → PyPDF2 → Text Extraction → Chunking → Embedding Generation → FAISS Index

2. QUERY PROCESSING
   User Question → Query Encoding → Vector Search → Re-ranking → Context Selection

3. ANSWER GENERATION
   Selected Context → Prompt Construction → LLM Generation → Citation Addition

4. QUALITY ASSURANCE
   Generated Answer → Relevance Check → Citation Validation → Confidence Scoring → Output
```

## 🧩 Component Details

### **Document Processing Layer**
```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: PDF Files (Technical Documentation)                    │
│     ↓                                                          │
│  PyPDF2 Text Extraction                                       │
│     ↓                                                          │
│  Text Cleaning & Normalization                                │
│     ↓                                                          │
│  Chunking Strategy:                                            │
│    • Size: 512 tokens                                         │
│    • Overlap: 50 tokens                                       │
│    • Method: Word-based sliding window                        │
│     ↓                                                          │
│  Metadata Attachment:                                          │
│    • Source file name                                          │
│    • Chunk ID                                                 │
│    • Document length                                           │
│     ↓                                                          │
│  Output: Processed Text Chunks                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Embedding & Indexing Layer**
```
┌─────────────────────────────────────────────────────────────────┐
│                   EMBEDDING & INDEXING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model: sentence-transformers/all-mpnet-base-v2                │
│    • Dimension: 768                                            │
│    • Quality: MTEB Score 63.3                                 │
│    • Speed: ~100ms per batch                                  │
│                                                                 │
│  FAISS Vector Store:                                           │
│    • Index Type: IndexFlatL2                                  │
│    • Distance Metric: L2 (Euclidean)                          │
│    • Search Time: ~5ms for 10K vectors                        │
│                                                                 │
│  Storage:                                                       │
│    • faiss.index (binary index file)                          │
│    • chunks.pkl (metadata & text)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Retrieval & Re-ranking Layer**
```
┌─────────────────────────────────────────────────────────────────┐
│                 RETRIEVAL & RE-RANKING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Semantic Search                                      │
│    • Method: Dense vector similarity                           │
│    • Candidates: Top-10 chunks                                │
│    • Speed: ~5ms                                              │
│                                                                 │
│  Stage 2: CrossEncoder Re-ranking                             │
│    • Model: cross-encoder/ms-marco-MiniLM-L6-v2               │
│    • Input: [query, context] pairs                            │
│    • Output: Relevance scores (0-1)                           │
│    • Selection: Top-3 highest scored                          │
│    • Speed: ~200ms                                            │
│                                                                 │
│  Quality Improvement: +15-20% precision                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Generation & Guardrails Layer**
```
┌─────────────────────────────────────────────────────────────────┐
│              GENERATION & GUARDRAILS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM: google/flan-t5-large                                    │
│    • Parameters: 780M                                          │
│    • Type: Instruction-tuned T5                               │
│    • Deployment: CPU-friendly                                  │
│    • Memory: ~4GB RAM requirement                             │
│                                                                 │
│  Prompt Engineering:                                           │
│    • Context injection (top-3 chunks)                         │
│    • Citation enforcement instructions                         │
│    • Fallback behavior specification                          │
│                                                                 │
│  Guardrails:                                                   │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ 1. Relevance Threshold (CrossEncoder > 0.5)        │   │
│    │ 2. Citation Requirement Enforcement                │   │
│    │ 3. Confidence Scoring (High/Medium/Low)            │   │
│    │ 4. Fallback: "Cannot find answer in documents"    │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Performance Characteristics

### **Latency Breakdown**
```
Total End-to-End Latency: ~800ms

┌─────────────────────────┬──────────┬─────────────┐
│ Component               │ Time     │ Percentage  │
├─────────────────────────┼──────────┼─────────────┤
│ Query Encoding          │ ~50ms    │ 6%          │
│ Vector Search (FAISS)   │ ~5ms     │ 1%          │
│ CrossEncoder Re-ranking │ ~200ms   │ 25%         │
│ LLM Generation          │ ~500ms   │ 62%         │
│ Post-processing         │ ~45ms    │ 6%          │
└─────────────────────────┴──────────┴─────────────┘
```

### **Memory Usage**
```
Total Memory Footprint: ~1.2GB

┌─────────────────────────┬──────────┬─────────────┐
│ Component               │ Memory   │ Percentage  │
├─────────────────────────┼──────────┼─────────────┤
│ Embedding Model         │ ~420MB   │ 35%         │
│ CrossEncoder            │ ~150MB   │ 13%         │
│ FLAN-T5 Large          │ ~500MB   │ 42%         │
│ FAISS Index            │ ~50MB    │ 4%          │
│ System Overhead        │ ~80MB    │ 7%          │
└─────────────────────────┴──────────┴─────────────┘
```

## 🚀 Deployment Architecture

### **Current (Prototype)**
```
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL DEPLOYMENT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   Single Node   │                                           │
│  │                 │                                           │
│  │  • Python App   │                                           │
│  │  • CPU Only     │                                           │
│  │  • Local Files  │                                           │
│  │  • 4GB RAM      │                                           │
│  │                 │                                           │
│  └─────────────────┘                                           │
│                                                                 │
│  Capacity: ~1K docs, Single user                              │
│  Cost: $0 (local machine)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Production (Phase 2)**
```
┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCTION DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Load Balancer│  │  App Node   │  │  App Node   │            │
│  │   (nginx)   │─▶│    (GPU)    │  │    (GPU)    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                           │               │                     │
│                           ▼               ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Redis     │  │ PostgreSQL  │  │   S3/MinIO  │            │
│  │  (Cache)    │  │ (+pgvector) │  │  (Docs)     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  Capacity: ~10K docs, 50 users                                │
│  Cost: ~$300-500/month                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Enterprise (Phase 3)**
```
┌─────────────────────────────────────────────────────────────────┐
│                  ENTERPRISE DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Kubernetes Cluster                     │   │
│  │                                                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │   API    │ │Embedding │ │   LLM    │ │ Vector   │  │   │
│  │  │ Gateway  │ │ Service  │ │ Service  │ │   DB     │  │   │
│  │  │          │ │          │ │          │ │(Weaviate)│  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  │                                                         │   │
│  │  Auto-scaling • Multi-region • A/B Testing            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Capacity: 100K+ docs, 100+ users                             │
│  Cost: ~$1.5K-3K/month                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Integration Points

### **Input Interfaces**
- REST API endpoints
- Python SDK
- Web UI (Gradio/Streamlit)
- CLI interface

### **Output Formats**
- JSON responses
- Structured citations
- Debug information
- Confidence metrics

### **Monitoring & Observability**
- Query latency tracking
- Relevance score analytics
- Usage pattern analysis
- Error rate monitoring

---

*This diagram represents the complete RAG system architecture with component interactions, data flow, and deployment considerations.*