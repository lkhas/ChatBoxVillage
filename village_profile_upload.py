"""
HR Policy Chatbot - File Upload and Knowledge Base Management
This module handles uploading and processing of HR policy documents (PDF, DOCX, TXT)
and creates a vector database for retrieval augmented generation (RAG).
"""

import os
import tempfile
import streamlit as st
from typing import List, Optional
from pathlib import Path
import shutil

# LangChain imports
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Additional imports for different file types
try:
    from langchain.document_loaders import UnstructuredWordDocumentLoader
except ImportError:
    UnstructuredWordDocumentLoader = None

class HRPolicyKnowledgeBase:
    """
    Manages the HR Policy Knowledge Base with file upload and vector database creation.
    """
    
    def __init__(self, openai_api_key: str, vector_store_path: str = "hr_policy_vectorstore"):
        """
        Initialize the HR Policy Knowledge Base.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            vector_store_path: Path to save/load the vector store
        """
        self.openai_api_key = openai_api_key
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        
        # Create directory for vector store if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Try to load existing vector store
        self.load_vectorstore()
    
    def load_vectorstore(self):
        """Load existing vector store if available."""
        try:
            if os.path.exists(self.vector_store_path) and os.listdir(self.vector_store_path):
                self.vectorstore = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success(f"Loaded existing knowledge base with {self.vectorstore.index.ntotal} documents")
        except Exception as e:
            st.warning(f"Could not load existing vector store: {str(e)}")
            self.vectorstore = None
    
    def save_vectorstore(self):
        """Save the vector store to disk."""
        if self.vectorstore:
            try:
                self.vectorstore.save_local(self.vector_store_path)
                st.success("Knowledge base saved successfully!")
            except Exception as e:
                st.error(f"Error saving knowledge base: {str(e)}")
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """
        Load a document based on its file type.
        
        Args:
            file_path: Path to the document
            file_type: Type of the document (pdf, docx, txt)
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            if file_type.lower() == 'pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            
            elif file_type.lower() in ['docx', 'doc']:
                try:
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                except Exception as e:
                    if UnstructuredWordDocumentLoader:
                        loader = UnstructuredWordDocumentLoader(file_path)
                        documents = loader.load()
                    else:
                        raise e
            
            elif file_type.lower() == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            
            else:
                st.error(f"Unsupported file type: {file_type}")
                return []
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_type': file_type,
                    'processed_at': str(pd.Timestamp.now())
                })
            
            return documents
            
        except Exception as e:
            st.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def process_uploaded_files(self, uploaded_files) -> bool:
        """
        Process uploaded files and add them to the knowledge base.
        
        Args:
            uploaded_files: List of uploaded files from Streamlit
            
        Returns:
            bool: True if processing was successful
        """
        if not uploaded_files:
            st.warning("No files uploaded")
            return False
        
        all_documents = []
        
        for uploaded_file in uploaded_files:
            # Get file extension
            file_extension = Path(uploaded_file.name).suffix.lower().replace('.', '')
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load document
                documents = self.load_document(tmp_file_path, file_extension)
                
                if documents:
                    all_documents.extend(documents)
                    st.success(f"Successfully loaded {uploaded_file.name} ({len(documents)} pages/sections)")
                else:
                    st.error(f"Failed to load {uploaded_file.name}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        if all_documents:
            return self.add_documents_to_vectorstore(all_documents)
        
        return False
    
    def add_documents_to_vectorstore(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            bool: True if successful
        """
        try:
            # Split documents into chunks
            text_chunks = self.text_splitter.split_documents(documents)
            
            if not text_chunks:
                st.error("No text chunks created from documents")
                return False
            
            # Create or update vector store
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(text_chunks, self.embeddings)
                st.success(f"Created new knowledge base with {len(text_chunks)} text chunks")
            else:
                self.vectorstore.add_documents(text_chunks)
                st.success(f"Added {len(text_chunks)} text chunks to existing knowledge base")
            
            # Save the updated vector store
            self.save_vectorstore()
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            st.error("No knowledge base available. Please upload documents first.")
            return []
        
        try:
            relevant_docs = self.vectorstore.similarity_search(query, k=k)
            return relevant_docs
        except Exception as e:
            st.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def get_knowledge_base_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        if not self.vectorstore:
            return {"total_documents": 0, "status": "No knowledge base loaded"}
        
        try:
            total_docs = self.vectorstore.index.ntotal
            return {
                "total_documents": total_docs,
                "status": "Knowledge base loaded",
                "vector_store_path": self.vector_store_path
            }
        except Exception as e:
            return {"total_documents": 0, "status": f"Error: {str(e)}"}
    
    def clear_knowledge_base(self):
        """Clear the entire knowledge base."""
        try:
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
                os.makedirs(self.vector_store_path, exist_ok=True)
            
            self.vectorstore = None
            st.success("Knowledge base cleared successfully!")
            
        except Exception as e:
            st.error(f"Error clearing knowledge base: {str(e)}")


def main():
    """Main function to run the HR Policy upload interface."""
    st.set_page_config(
        page_title="HR Policy Knowledge Base Manager",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 HR Policy Knowledge Base Manager")
    st.markdown("Upload and manage HR policy documents for the chatbot knowledge base.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # OpenAI API Key input
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for embeddings"
    )
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    
    # Initialize knowledge base
    try:
        kb = HRPolicyKnowledgeBase(openai_api_key)
    except Exception as e:
        st.error(f"Error initializing knowledge base: {str(e)}")
        st.stop()
    
    # Display knowledge base stats
    stats = kb.get_knowledge_base_stats()
    st.sidebar.metric("Documents in Knowledge Base", stats["total_documents"])
    st.sidebar.info(stats["status"])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload HR Policy Documents")
        
        # File upload widget
        uploaded_files = st.file_uploader(
            "Choose HR policy files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload PDF, DOCX, DOC, or TXT files containing HR policies"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # Size in KB
                st.write(f"- {file.name} ({file_size:.1f} KB)")
            
            if st.button("Process and Add to Knowledge Base", type="primary"):
                with st.spinner("Processing uploaded files..."):
                    success = kb.process_uploaded_files(uploaded_files)
                    if success:
                        st.rerun()  # Refresh to show updated stats
        
        # Search functionality
        st.header("Test Knowledge Base Search")
        search_query = st.text_input("Enter a search query to test the knowledge base:")
        
        if search_query and st.button("Search"):
            with st.spinner("Searching knowledge base..."):
                results = kb.search_knowledge_base(search_query)
                
                if results:
                    st.success(f"Found {len(results)} relevant documents")
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"Result {i} - {doc.metadata.get('source_file', 'Unknown')}"):
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
                else:
                    st.info("No relevant documents found.")
    
    with col2:
        st.header("Management")
        
        # Clear knowledge base button
        if st.button("Clear Knowledge Base", type="secondary"):
            if st.confirm("Are you sure you want to clear the entire knowledge base?"):
                kb.clear_knowledge_base()
                st.rerun()
        
        # Download vector store (placeholder)
        st.header("Export")
        st.info("Knowledge base is automatically saved locally in the 'hr_policy_vectorstore' directory.")
        
        # Instructions
        st.header("Instructions")
        st.markdown("""
        1. **Upload Documents**: Select PDF, DOCX, DOC, or TXT files containing HR policies
        2. **Process**: Click "Process and Add to Knowledge Base" to add documents
        3. **Test**: Use the search function to verify documents are properly indexed
        4. **Clear**: Use "Clear Knowledge Base" to remove all documents and start fresh
        
        **Supported Formats:**
        - PDF (.pdf)
        - Word Documents (.docx, .doc)  
        - Text Files (.txt)
        """)


if __name__ == "__main__":
    import pandas as pd
    main()