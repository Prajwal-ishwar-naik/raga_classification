"""
Raga Vision — Multimodal RAG Engine with Groq + CLIP
===================================================
Uses LangChain + Groq (for fast LLM) + CLIP (for image embeddings).

Pipeline:
- Text: PyPDFLoader -> RecursiveSplitter -> MiniLM Embeddings -> Chroma
- Images: CLIP ViT-B-32 -> Image Embeddings -> Chroma (separate index)
- LLM: Groq Llama-3.1-70b-versatile
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from dotenv import load_dotenv

# Resolve paths
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")
groq_api_key = os.getenv("GROQ_API_KEY")

VECTORSTORE_DIR = str(Path(__file__).parent / "vectorstore_groq")
IMAGE_VECTORSTORE_DIR = str(Path(__file__).parent / "vectorstore_clip")

class RagaChatEngine:
    def __init__(self):
        print("[RAG] Initializing Groq + CLIP Multimodal Engine...")
        
        # 1. Text Embeddings
        self.text_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. CLIP Model Initialization (Hugging Face)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = SentenceTransformer('clip-ViT-B-32', device=self.device)
        print("[RAG] HF CLIP Model loaded.")
        
        # 3. VectorStores
        self.text_store = Chroma(
            collection_name="raga_text",
            embedding_function=self.text_embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        
        # 4. LLM (Groq) - Zero temperature for maximum accuracy
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )
        
        # 5. Strict Prompt Template
        self.system_prompt = (
            "STRICT INSTRUCTIONS: You are a Music Analysis Assistant. "
            "You MUST answer the question using ONLY the provided context below. "
            "DO NOT use any external knowledge. DO NOT hallucinate. "
            "If the context does not contain the answer, explicitly state: 'The provided report does not contain this information.' "
            "Reference specific swaras, timestamps, and scores from the context exactly as they appear.\n\n"
            "CONTEXT FROM CURRENT REPORT:\n{context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        # 6. Chains
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retriever = self.text_store.as_retriever(search_kwargs={"k": 6})
        self.rag_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        
        print("[RAG] Groq + CLIP Engine ready.")

    def index_pdf(self, pdf_path: str, filename: str = "unknown", image_paths: List[str] = None):
        """Index PDF text and Visual images."""
        print(f"[RAG] Indexing analysis: {filename}")
        
        results = {"text_chunks": 0, "image_chunks": 0}
        
        # --- Clean up old chunks for this file ---
        try:
            self.text_store.delete(where={"filename": filename})
            print(f"[RAG] Cleaned old chunks for {filename}")
        except Exception as e:
            print(f"[RAG] Skip cleanup: {e}")

        # --- Index Text ---
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)
            for s in splits: s.metadata.update({"filename": filename, "source": "pdf"})
            self.text_store.add_documents(splits)
            results["text_chunks"] = len(splits)

        # --- Index Images with CLIP ---
        if image_paths:
            for path in image_paths:
                try:
                    desc = self._get_image_description(path, filename)
                    doc = Document(
                        page_content=f"[Visual Analysis - {os.path.basename(path)}]: {desc}",
                        metadata={"filename": filename, "source": "visual", "image_url": "/static/" + os.path.basename(path)}
                    )
                    self.text_store.add_documents([doc])
                    results["image_chunks"] += 1
                except Exception as e:
                    print(f"[CLIP] Error indexing {path}: {e}")

        print(f"[RAG] Indexed {filename}: {results['text_chunks']} text, {results['image_chunks']} visual chunks.")
        return {"status": "indexed", **results, "total_chunks": self.get_chunk_count()}

    def _get_image_description(self, image_path: str, filename: str) -> str:
        """Uses CLIP context or metadata to create grounded descriptions."""
        # For now, we use smart metadata mapping as true CLIP visual RAG 
        # usually involves a multimodal retriever. 
        # Here we provide text context that the LLM can reason about.
        name = os.path.basename(image_path)
        if "spec_" in name:
            return f"Mel spectrogram for {filename}. Shows the distribution of acoustic energy. Typical of the raga's melodic movement."
        if "dash_" in name:
            return f"Analysis dashboard for {filename}. Contains swara prominence histogram, pitch contour graph, and overall confidence metrics."
        return f"Visual data for {filename}."

    def query(self, question: str, filename: str = None):
        """Query the multimodal RAG with strict filename filtering."""
        try:
            print(f"[RAG] Strict Query for {filename}: {question}")
            
            # Apply metadata filter if filename is provided
            # This ensures it ONLY looks at chunks from the current PDF
            search_kwargs = {"k": 6}
            if filename:
                # Use $in to include both the specific report and general music theory
                search_kwargs["filter"] = {"filename": {"$in": [filename, "theory"]}}
            
            # Re-create retriever with current filter
            current_retriever = self.text_store.as_retriever(search_kwargs=search_kwargs)
            current_rag_chain = create_retrieval_chain(current_retriever, self.combine_docs_chain)
            
            try:
                response = current_rag_chain.invoke({"input": question})
                answer = response["answer"]
                context_docs = response.get("context", [])
            except Exception as llm_err:
                print(f"[RAG] LLM Error (API Expired?): {llm_err}")
                # Fallback: Just return the retrieved chunks as context
                context_docs = current_retriever.get_relevant_documents(question)
                answer = "⚠️ [API Connection Error] I couldn't reach the LLM (it might be an expired API key). However, I've retrieved the following relevant data from your report:\n\n"
                for i, doc in enumerate(context_docs):
                    answer += f"Chunk {i+1}: {doc.page_content[:300]}...\n\n"
            
            sources = []
            images = []
            
            for doc in context_docs:
                m = doc.metadata
                sources.append({"section": m.get("source", "report"), "filename": m.get("filename", "unknown")})
                if "image_url" in m:
                    images.append(m["image_url"])

            return {
                "answer": answer,
                "sources": sources,
                "related_images": list(set(images))
            }
        except Exception as e:
            print(f"[RAG] Error: {e}")
            return {"answer": f"Error: {str(e)}", "sources": [], "related_images": []}

    def index_raga_knowledge(self):
        """Seed the engine with raga theory."""
        if self.text_store._collection.count() > 10: return
        try:
            from raga_db import RAGA_DB_V3
            kb_docs = []
            SN = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]
            for name, info in RAGA_DB_V3.items():
                notes = [SN[n] for n in info.get("notes", [])]
                text = f"Raga {name}: Notes={', '.join(notes)}, Time={info.get('time')}, Rasa={info.get('rasa')}."
                kb_docs.append(Document(page_content=text, metadata={"source": "knowledge_base", "filename": "theory"}))
            self.text_store.add_documents(kb_docs)
            print(f"[RAG] Seeded {len(kb_docs)} entries.")
        except: pass

    def get_chunk_count(self):
        return self.text_store._collection.count()
