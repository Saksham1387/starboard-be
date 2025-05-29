from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import uuid
from pathlib import Path
import logging
from urllib.parse import urlparse
from utils import download_file_from_url, setup_llm_with_source_prompt,setup_llm_with_summary_prompt,system_prompt,convert_string_to_json
# Your existing imports
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownNodeParser

from dataclasses import dataclass
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Service API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for RAG systems
rag_systems: Dict[str, Any] = {}
processing_status: Dict[str, Dict] = {}

# Updated Pydantic models
class ProcessS3Request(BaseModel):
    user_id: str
    file_urls: List[str]
    file_names: Optional[List[str]] = None  # Optional custom names for files

class ProcessResponse(BaseModel):
    project_id: str
    status: str
    message: str

class ChatRequest(BaseModel):
    project_id: str
    message: str
    user_id: str

class ChatResponse(BaseModel):
    project_id: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SummaryRequest(BaseModel):
    project_id: str
    user_id: str
    max_length: Optional[int] = 500  # Optional parameter to control summary length

class SummaryResponse(BaseModel):
    project_id: str
    summary: Dict[str, Any]
    metadata: Dict[str, Any]

class StatusResponse(BaseModel):
    project_id: str
    status: str
    progress: float
    message: str
    error: Optional[str] = None

# Your existing source tracking classes
@dataclass
class SourceInfo:
    file_name: str
    page_number: int = None
    chunk_id: str = None
    similarity_score: float = None
    text_snippet: str = None

class SourceTrackingQueryEngine:
    def __init__(self, query_engine, include_sources=True):
        self.query_engine = query_engine
        self.include_sources = include_sources
        self.last_sources = []
    
    def query(self, question: str):
        response = self.query_engine.query(question)
        
        if hasattr(self.query_engine, 'retriever'):
            nodes = self.query_engine.retriever.retrieve(question)
            self.last_sources = self._extract_sources(nodes)
        
        return response
    
    def _extract_sources(self, nodes) -> List[SourceInfo]:
        sources = []
        for i, node in enumerate(nodes):
            source_info = SourceInfo(
                file_name=node.metadata.get('file_name', 'Unknown'),
                page_number=node.metadata.get('page_label', None),
                chunk_id=f"chunk_{i}",
                similarity_score=node.score if hasattr(node, 'score') else None,
                text_snippet=node.text[:200] + "..." if len(node.text) > 200 else node.text
            )
            sources.append(source_info)
        return sources
    
    def get_last_sources(self) -> List[Dict]:
        return [
            {
                "file_name": source.file_name,
                "page_number": source.page_number,
                "chunk_id": source.chunk_id,
                "similarity_score": source.similarity_score,
                "text_snippet": source.text_snippet
            }
            for source in self.last_sources
        ]

def create_user_rag_system(project_id: str, user_id: str, data_dir: str):
    """Create RAG system for specific user project"""
    try:
        # Set up LLM and embeddings
        Settings.llm = setup_llm_with_source_prompt()
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",  
            dimensions=1536
        )
        
        # Load documents
        logger.info(f"Loading documents from {data_dir}")
        documents = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=[".md", ".txt", ".pdf", ".csv"],  
            recursive=True
        ).load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {data_dir}")
        
        # Enhance metadata with user/project info
        for doc in documents:
            if 'file_name' not in doc.metadata:
                doc.metadata['file_name'] = doc.metadata.get('filename', 'Unknown')
            
            # Add user isolation metadata
            doc.metadata['user_id'] = user_id
            doc.metadata['project_id'] = project_id
            doc.metadata['doc_id'] = hash(doc.text[:100])
        
        # Create user-specific ChromaDB collection
        db_path = f"./chroma_db_{project_id}"
        db = chromadb.PersistentClient(path=db_path)
        
        # Collection name includes project for isolation
        collection_name = f"project_{project_id}"
        chroma_collection = db.get_or_create_collection(
            name=collection_name,
            metadata={
                "user_id": user_id,
                "project_id": project_id
            }
        )
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        node_parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True
        )
        
        # Create vector index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            node_parser=node_parser,
            show_progress=True
        )
        
        # Set up retriever and query engine
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=False,
            streaming=False
        )
        
        base_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.2)
            ],
        )
        
        # Wrap with source tracking
        source_tracking_engine = SourceTrackingQueryEngine(base_query_engine)
        
        return source_tracking_engine
        
    except Exception as e:
        logger.error(f"Error creating RAG system: {str(e)}")
        raise

async def process_documents_background(project_id: str, user_id: str, file_urls: List[str], file_names: Optional[List[str]] = None):
    """Background task to process documents from URLs"""
    try:
        # Update status to processing
        processing_status[project_id] = {
            "status": "processing",
            "progress": 0.1,
            "message": "Starting document download and processing..."
        }
        
        # Create upload directory
        upload_dir = f"./uploads/{project_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Download files from URLs
        downloaded_files = []
        total_files = len(file_urls)
        
        for i, url in enumerate(file_urls):
            try:
                processing_status[project_id]["progress"] = 0.1 + (0.3 * i / total_files)
                processing_status[project_id]["message"] = f"Downloading file {i+1}/{total_files}..."
                
                filename = download_file_from_url(url, upload_dir)
                
                # Use custom filename if provided
                if file_names and i < len(file_names) and file_names[i]:
                    old_path = os.path.join(upload_dir, filename)
                    new_filename = file_names[i]
                    new_path = os.path.join(upload_dir, new_filename)
                    os.rename(old_path, new_path)
                    downloaded_files.append(new_filename)
                else:
                    downloaded_files.append(filename)
                    
            except Exception as e:
                logger.error(f"Failed to download file from {url}: {str(e)}")
                processing_status[project_id] = {
                    "status": "error",
                    "progress": 0.0,
                    "message": f"Failed to download file from {url}",
                    "error": str(e)
                }
                return
        
        # Update status to processing embeddings
        processing_status[project_id]["progress"] = 0.5
        processing_status[project_id]["message"] = "Creating embeddings..."
        
        # Create RAG system
        rag_system = create_user_rag_system(project_id, user_id, upload_dir)
        
        # Store RAG system
        rag_systems[project_id] = {
            "rag_engine": rag_system,
            "user_id": user_id,
            "created_at": str(uuid.uuid4()),
            "downloaded_files": downloaded_files
        }
        
        # Clean up: Delete the upload directory after successful processing
        try:
            shutil.rmtree(upload_dir)
            logger.info(f"Successfully cleaned up upload directory for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up upload directory for project {project_id}: {str(e)}")
        
        # Update status to completed
        processing_status[project_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": f"Document processing completed successfully! Downloaded and processed {len(downloaded_files)} files."
        }
        
        logger.info(f"RAG system created successfully for project {project_id} with {len(downloaded_files)} files")
        
    except Exception as e:
        # Clean up upload directory even if processing failed
        try:
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                logger.info(f"Cleaned up upload directory after failed processing for project {project_id}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up upload directory after error for project {project_id}: {str(cleanup_error)}")
            
        processing_status[project_id] = {
            "status": "error",
            "progress": 0.0,
            "message": "Document processing failed",
            "error": str(e)
        }
        logger.error(f"Error processing documents for project {project_id}: {str(e)}")

@app.post("/process", response_model=ProcessResponse)
async def process_documents_from_urls(
    background_tasks: BackgroundTasks,
    request: ProcessS3Request
):
    """
    Process documents from S3 URLs or any accessible URLs
    """
    try:
        # Validate input
        if not request.file_urls:
            raise HTTPException(status_code=400, detail="No file URLs provided")
        
        if len(request.file_urls) > 50:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files allowed")
        
        # Validate URLs
        for url in request.file_urls:
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
        
        # Generate unique project ID
        project_id = str(uuid.uuid4())
        
        # Initialize processing status
        processing_status[project_id] = {
            "status": "initialized",
            "progress": 0.0,
            "message": f"Initialized processing for {len(request.file_urls)} files"
        }
        
        # Start background processing
        background_tasks.add_task(
            process_documents_background, 
            project_id, 
            request.user_id, 
            request.file_urls,
            request.file_names
        )
        
        return ProcessResponse(
            project_id=project_id,
            status="processing",
            message=f"Processing started for {len(request.file_urls)} files from URLs"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{project_id}", response_model=StatusResponse)
async def get_processing_status(project_id: str):
    """
    Get processing status for a project
    """
    if project_id not in processing_status:
        raise HTTPException(status_code=404, detail="Project not found")
    
    status_info = processing_status[project_id]
    return StatusResponse(
        project_id=project_id,
        status=status_info["status"],
        progress=status_info["progress"],
        message=status_info["message"],
        error=status_info.get("error")
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """
    Chat with processed documents
    """
    try:
        # Check if project exists and is ready
        if request.project_id not in rag_systems:
            # Check if still processing
            if (request.project_id in processing_status and 
                processing_status[request.project_id]["status"] == "processing"):
                raise HTTPException(
                    status_code=202, 
                    detail="Documents are still being processed. Please wait."
                )
            else:
                raise HTTPException(
                    status_code=404, 
                    detail="Project not found or processing failed"
                )
        
        # Verify user ownership
        project_info = rag_systems[request.project_id]
        if project_info["user_id"] != request.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get RAG engine
        rag_engine = project_info["rag_engine"]
        
        # Process query
        logger.info(f"Processing chat query for project {request.project_id}")
        response = rag_engine.query(request.message)
        sources = rag_engine.get_last_sources()

        print("Response from LLM:", response)
        
        
        # Format response
        result = {
            "question": request.message,
            "answer": str(response), 
            "sources": sources,
            "metadata": {
                "source_count": len(sources),
                "avg_similarity": sum(s["similarity_score"] or 0 for s in sources) / len(sources) if sources else 0,
                "unique_files": list(set(s["file_name"] for s in sources))
            }
        }
        
        return ChatResponse(
            project_id=request.project_id,
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate a summary of the processed documents
    """
    try:
        # Check if project exists and is ready
        if request.project_id not in rag_systems:
            # Check if still processing
            if (request.project_id in processing_status and 
                processing_status[request.project_id]["status"] == "processing"):
                raise HTTPException(
                    status_code=202, 
                    detail="Documents are still being processed. Please wait."
                )
            else:
                raise HTTPException(
                    status_code=404, 
                    detail="Project not found or processing failed"
                )
        
        # Verify user ownership
        project_info = rag_systems[request.project_id]
        if project_info["user_id"] != request.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get RAG engine
        rag_engine = project_info["rag_engine"]
        
        # Set up summary-specific LLM
        Settings.llm = setup_llm_with_summary_prompt()
        
        # Generate summary using a specific prompt
        summary_prompt = system_prompt
        response = rag_engine.query(summary_prompt)
        sources = rag_engine.get_last_sources()
        
        print("Response from LLM:", response)
        # cleaned_json_str = str(response).encode('utf-8').decode('unicode_escape')
    
        # # Step 2: Parse it into a dictionary
        # parsed_data = json.loads(cleaned_json_str)

        parsed_json = convert_string_to_json(str(response))
        
        # Format response
        result = {
            "summary": parsed_json,
            "metadata": {
                "source_count": len(sources),
                "unique_files": list(set(s["file_name"] for s in sources)),
                "summary_length": len(str(response).split())
            }
        }
        
        return SummaryResponse(
            project_id=request.project_id,
            summary=result["summary"],
            metadata=result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summary endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Verify required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)