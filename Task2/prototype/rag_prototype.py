"""
RAG SYSTEM PROTOTYPE FOR TECHNICAL DOCUMENTATION (PRO VERSION)
=================================================
A high-performance RAG system using upgraded open-source tools:
- Document ingestion from PDFs
- FAISS vector store for fast retrieval
- Sentence Transformers (all-mpnet-base-v2) for high-quality embeddings
- Cross-Encoder Re-ranking for high retrieval precision
- Hugging Face FLAN-T5-Large for powerful LLM generation
- Citation enforcement, guardrails, and automated evaluation.

Author: Tharun Kumar (Updated)
Date: 2025
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import re
import warnings
import numpy as np
import pickle
import faiss
import pandas as pd # For evaluation
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader

# Vector store, Embeddings, and Reranking
from sentence_transformers import SentenceTransformer, CrossEncoder

# LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully! (Including pandas and CrossEncoder)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
DOCS_FOLDER = "docs/" 
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # UPGRADE: Better performance
LLM_MODEL = "google/flan-t5-large"                          # UPGRADE: More capable LLM
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"       # NEW: For high precision retrieval

CHUNK_SIZE = 512    # tokens
CHUNK_OVERLAP = 50  # tokens
TOP_K_RETRIEVAL = 10 # Retrieve 10 candidates first
TOP_K_RERANK = 3     # Select the best 3 after re-ranking
MAX_ANSWER_LENGTH = 250  # Max tokens in answer

# Create folders
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs("rag_index", exist_ok=True)

print(f"📂 Documents folder: {DOCS_FOLDER}")
print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
print(f"🔄 Reranker model: {RERANKER_MODEL}")
print(f"💬 LLM model: {LLM_MODEL}")
print("="*80)

# ============================================================================
# DOCUMENT INGESTION & PREPROCESSING
# ============================================================================
class DocumentProcessor:
    """Extracts and preprocesses text from PDFs"""
    
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                # Simple way to track page, but chunking is word-based here
                text += f"\n[DOCUMENT_PAGE_{page_num+1}]\n"
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"⚠️  Error reading {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Keep essential punctuation and document page markers
        text = re.sub(r'[^\w\s.,!?;:()\[\]\-]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks (word count based)"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Simple metadata extraction for demo - better methods exist (e.g., LlamaIndex)
            chunks.append({
                'text': chunk_text,
                'metadata': metadata,
                'chunk_id': len(chunks)
            })
            
        return chunks
    
    def process_documents(self, folder_path: str) -> List[Dict]:
        """Process all PDFs in folder"""
        all_chunks = []
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        print(f"\n📚 Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"  Processing: {pdf_file}")
            
            # Extract and clean
            raw_text = self.extract_text_from_pdf(pdf_path)
            clean_text = self.clean_text(raw_text)
            
            # Create chunks with metadata
            metadata = {
                'source': pdf_file,
                'doc_length': len(clean_text)
            }
            chunks = self.chunk_text(clean_text, metadata)
            all_chunks.extend(chunks)
            
            print(f"    ✅ Created {len(chunks)} chunks")
            
        print(f"\n✅ Total chunks created: {len(all_chunks)}")
        return all_chunks

# ============================================================================
# VECTOR STORE & RETRIEVAL
# ============================================================================
class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_model_name: str):
        print(f"\n🔧 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Embedding dimension: {self.dimension}")
    
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from document chunks"""
        print(f"\n🏗️  Building FAISS index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print("  Generating embeddings...")
        # Use a higher batch size if memory allows, but default is fine for demo
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Search for most relevant chunks using semantic similarity"""
        if self.index is None:
            raise ValueError("Index not built yet!")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return chunks with scores (distance is used here, lower is better)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.chunks[idx], float(dist)))
        
        return results
    
    def save_index(self, path: str = "rag_index/"):
        """Save index and chunks to disk"""
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Index saved to {path}")
    
    def load_index(self, path: str = "rag_index/"):
        """Load index and chunks from disk"""
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"✅ Index loaded from {path}")

# ============================================================================
# LLM GENERATION WITH GUARDRAILS AND RERANKING
# ============================================================================
class RAGGenerator:
    """Generate answers using retrieved context with citations and reranking"""
    
    def __init__(self, model_name: str, reranker_model_name: str):
        # LLM Setup
        print(f"\n🤖 Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Use a Text2Text Generation Pipeline (suitable for T5)
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=MAX_ANSWER_LENGTH
        )
        print("✅ LLM loaded successfully")
        
        # Reranker Setup
        print(f"🔄 Loading Reranker: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)
        print("✅ Reranker loaded successfully")
    
    def rerank_contexts(self, query: str, contexts: List[Tuple[Dict, float]], top_k_rerank: int) -> List[Tuple[Dict, float]]:
        """Re-rank contexts using a CrossEncoder model for improved precision"""
        if not contexts:
            return []

        # Prepare pairs: [[query, context_text_1], [query, context_text_2], ...]
        query_and_contexts = [[query, chunk['text']] for chunk, _ in contexts]
        
        # Get scores (higher is better for CrossEncoder)
        reranker_scores = self.reranker.predict(query_and_contexts)
        
        # Combine chunks with new scores and sort by reranker score
        reranked_tuples = sorted(
            zip([chunk for chunk, _ in contexts], reranker_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return only the top N chunks and their scores
        return [(chunk, float(score)) for chunk, score in reranked_tuples[:top_k_rerank]]

    def create_prompt(self, query: str, contexts: List[Tuple[Dict, float]]) -> str:
        """Create prompt with retrieved and ranked context"""
        prompt = "Answer the following question using ONLY the provided context. "
        prompt += "If the answer is not in the context, clearly state that 'I cannot find the answer in the documents.'\n\n"
        
        prompt += "Context:\n"
        # The contexts here are already the top-k reranked results
        for i, (chunk, score) in enumerate(contexts):
            # Show reranker score for transparency
            prompt += f"[{i+1}] Source: {chunk['metadata']['source']} (Relevance Score: {score:.3f})\n"
            # Limit context text length in the prompt to conserve tokens
            prompt += f"{chunk['text'][:400]}...\n\n"
        
        prompt += f"Question: {query}\n"
        prompt += "Answer (must include source citation [1], [2], etc.):"
        
        return prompt
    
    def generate_answer(self, query: str, contexts: List[Tuple[Dict, float]]) -> Dict:
        """Generate answer with guardrails"""
        
        # Reranked contexts are now passed in from the RAGSystem.query
        
        # Guardrail: Check if contexts are relevant (using the CrossEncoder score)
        # Assuming the first context has the highest score after reranking
        reranker_score = contexts[0][1] if contexts else 0.0

        if not contexts or reranker_score < 0.5: # 0.5 is a standard relevance threshold
            return {
                'answer': "I cannot find relevant information in the documents to answer this question. The retrieved context was not relevant enough.",
                'sources': [],
                'confidence': 'low'
            }
        
        # Create prompt
        prompt = self.create_prompt(query, contexts)
        
        # Generate
        try:
            # We use a simple generation approach for this T5 demo
            response = self.generator(prompt, max_length=MAX_ANSWER_LENGTH, do_sample=False, temperature=0.1)[0]['generated_text']
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'confidence': 'error'
            }
        
        # Extract sources
        sources = [chunk['metadata']['source'] for chunk, _ in contexts]
        
        return {
            'answer': response,
            'sources': list(set(sources)),
            'confidence': 'high' if reranker_score > 0.8 else 'medium'
        }

# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================
class RAGSystem:
    """Complete RAG system orchestrator"""
    
    def __init__(self):
        self.processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
        self.vector_store = VectorStore(EMBEDDING_MODEL)
        self.generator = RAGGenerator(LLM_MODEL, RERANKER_MODEL)
        self.is_indexed = False
        
    def ingest_documents(self, docs_folder: str):
        """Ingest and index documents"""
        print("\n" + "="*80)
        print("📥 DOCUMENT INGESTION")
        print("="*80)
        
        # Process documents
        chunks = self.processor.process_documents(docs_folder)
        
        # Build index
        self.vector_store.build_index(chunks)
        
        # Save for future use
        self.vector_store.save_index()
        
        self.is_indexed = True
        print("\n✅ Documents ingested and indexed successfully!")
    
    def query(self, question: str, top_k_retrieval: int = TOP_K_RETRIEVAL, top_k_rerank: int = TOP_K_RERANK) -> Dict:
        """Query the RAG system, incorporating re-ranking"""
        if not self.is_indexed:
            return {
                'answer': "System not ready. Please ingest documents first.",
                'sources': [],
                'confidence': 'error'
            }
        
        # 1. Retrieve a large number of candidates (e.g., 10)
        candidate_contexts = self.vector_store.search(question, top_k_retrieval)
        
        # 2. Re-rank candidates to select the most relevant subset (e.g., 3)
        reranked_contexts = self.generator.rerank_contexts(question, candidate_contexts, top_k_rerank)
        
        # 3. Generate answer using the highly-relevant context
        result = self.generator.generate_answer(question, reranked_contexts)
        
        # Add retrieved chunks for transparency
        result['retrieved_chunks'] = [
            {
                'text': chunk['text'][:100] + "...",
                'source': chunk['metadata']['source'],
                'reranker_score': score
            }
            for chunk, score in reranked_contexts
        ]
        
        return result
    
    def run_evaluation(self, eval_path: str = "eval_questions.csv", output_path: str = "evaluation.csv"):
        """
        Runs an evaluation against a set of questions and saves results to a CSV.
        This provides a quantitative assessment of the RAG pipeline.
        """
        print("\n" + "="*80)
        print("📊 RUNNING EVALUATION")
        print("="*80)

        if not self.is_indexed:
            print("❌ Cannot run evaluation: Documents not indexed.")
            return

        try:
            eval_df = pd.read_csv(eval_path)
        except FileNotFoundError:
            print(f"❌ Evaluation file '{eval_path}' not found.")
            print("Please create this file using the provided template and add it to the script directory.")
            return
        except Exception as e:
            print(f"❌ Error reading evaluation file: {e}")
            return

        print(f"📝 Evaluating {len(eval_df)} questions...")
        results = []
        
        for index, row in eval_df.iterrows():
            question = row['question']
            ground_truth = row['ground_truth_answer']
            expected_sources = str(row['relevant_sources']).split(';') # Semi-colon separated sources
            
            rag_result = self.query(question)
            
            # Simple manual metrics: Check if the ground truth answer is present in RAG output (rough Factual Accuracy)
            is_accurate = ground_truth.lower() in rag_result['answer'].lower()
            
            # Check if all expected sources were cited by the LLM (Context Recall)
            cited_sources_str = ', '.join(rag_result['sources'])
            is_cited = all(source.strip() in cited_sources_str for source in expected_sources if source.strip() != 'nan')
            
            # Capture the results
            results.append({
                'question': question,
                'ground_truth_answer': ground_truth,
                'rag_answer': rag_result['answer'],
                'cited_sources': cited_sources_str,
                'expected_sources': ', '.join(expected_sources),
                'confidence': rag_result['confidence'],
                'is_accurate_check': is_accurate,
                'is_cited_check': is_cited
            })

        # Save to a new CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Evaluation complete! Results saved to '{output_path}'.")
        print("Inspect the CSV for detailed results.")
        
    
    def interactive_mode(self):
        """Interactive Q&A session"""
        print("\n" + "="*80)
        print("💬 INTERACTIVE MODE")
        print("="*80)
        print(f"Ask questions about your documents (using {LLM_MODEL}). Type 'quit' to exit.\n")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Don't forget to check your evaluation.csv!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching and Re-ranking...")
            result = self.query(question)
            
            print(f"\n💡 Answer ({result['confidence']} confidence):")
            print(f"   {result['answer']}\n")
            
            print(f"📚 Sources: {', '.join(result['sources'])}")
            print("\n--- Top Retrieved Contexts (for debugging) ---")
            for i, chunk in enumerate(result['retrieved_chunks']):
                print(f"[{i+1}] Score: {chunk['reranker_score']:.3f} | Source: {chunk['source']}")
                print(f"    Text: {chunk['text']}")
            print("-" * 80 + "\n")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("RAG SYSTEM PROTOTYPE (PRO) - UPGRADED")
    print("🚀" * 40)
    
    # Initialize system
    rag = RAGSystem()
    
    # Check if we need to ingest documents
    if not os.path.exists("rag_index/faiss.index") or len([f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]) != len(rag.vector_store.chunks):
        
        pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]
        
        if len(pdf_files) == 0:
            print(f"\n⚠️  No PDF files found in {DOCS_FOLDER}")
            print("Please add PDF files to the 'docs/' folder to start ingestion.")
            exit()
        
        rag.ingest_documents(DOCS_FOLDER)
    else:
        print("\n✅ Loading existing index...")
        rag.vector_store.load_index()
        rag.is_indexed = True
    
    # Run a quick example query
    print("\n" + "="*80)
    print("📝 EXAMPLE QUERY")
    print("="*80)
    q = "What are the common troubleshooting steps for system temperature anomaly?"
    print(f"❓ Question: {q}")
    result = rag.query(q)
    print(f"💡 Answer: {result['answer'][:150]}...")
    print(f"📚 Sources: {', '.join(result['sources'])}")
    print("-" * 80)
    
    # Run evaluation
    rag.run_evaluation()

    # Start interactive mode
    print("\n" + "="*80)
    user_input = input("\nStart interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.interactive_mode()
    
    print("\n✅ RAG System Demo Complete!")
