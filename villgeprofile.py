import streamlit as st
import openai
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader,UnstructuredWordDocumentLoader
from langchain.schema import Document
import os
import tempfile
import pandas as pd
from typing import List, Tuple
import io
import json

from dotenv import load_dotenv
load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")

# Securely read API key
openai.api_key = st.secrets["openai"]["api_key"]
st.write("API key loaded:", "api_key" in st.secrets["openai"])


# Set page configuration
st.set_page_config(
    page_title="Village Profile Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #234c6b;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #544056;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    
    .upload-section {
        background-color: #3f4a54;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        margin: 1rem 0;
    }
    
    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables for conversation history and file management"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text content"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        documents = []
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        elif file_extension == 'txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
        elif file_extension == 'csv':
            # Handle CSV files
            df = pd.read_csv(uploaded_file)
            text_content = df.to_string()
            documents = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]
        elif file_extension == 'docx':
            # Handle Word documents
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
            documents = loader.load()
        else:
            # Handle other text files
            content = uploaded_file.read().decode('utf-8')
            documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return []

def create_vector_store(documents, openai_api_key):
    """Create vector store from uploaded documents"""
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split all documents
        split_documents = []
        for doc in documents:
            splits = text_splitter.split_documents([doc])
            split_documents.extend(splits)
        
        if not split_documents:
            return None
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create vector store
        vector_store = FAISS.from_documents(split_documents, embeddings)
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def build_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    """Convert chat history tuples to message objects for LLM"""
    messages = []
    for human_msg, ai_msg in chat_history:
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))
    return messages

def create_conversational_chain(vector_store, openai_api_key):
    """Create conversational retrieval chain with chat history awareness"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        openai_api_key=openai_api_key
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": 5}
    )
    
    # Contextualize question prompt for history-aware retrieval
    contextualize_q_system_prompt = """  
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Answer question prompt
    qa_system_prompt = """
    You are a dedicated research support system for Indian government officials, designed to analyze and interpret official documents and administrative data.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise but comprehensive.
    Always reference the source document when possible.
    
    Context: {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def query_hr_policy(prompt: str, chat_history: List[Tuple[str, str]]) -> str:
    """Query the HR policy chatbot with conversation history"""
    
    if st.session_state.qa_chain is None:
        return "Please configure the chatbot first by providing your OpenAI API key and uploading knowledge documents."
    
    try:
        # Build chat history for LLM
        formatted_chat_history = build_chat_history(chat_history)
        
        # Get response from the chain
        response = st.session_state.qa_chain.invoke({
            "input": prompt,
            "chat_history": formatted_chat_history
        })
        
        return response["answer"]
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

def display_chat_history():
    """Display the chat history in the UI"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    
    # Page header
    st.markdown('<div class="main-header"><h1>🤖 Village Chatbot</h1><p>Upload your villgae documents and ask questions!</p></div>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
   
        
        # # OpenAI API Key input
        # openai_api_key = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     value=st.session_state.get("openai_api_key", ""),
        #     help="Enter your OpenAI API key to enable the chatbot"
        # )
        
        # if openai_api_key:
        #     st.session_state.openai_api_key = openai_api_key
        #     st.success("✅ API Key configured")

        openai_api_key = os.getenv("OPENAI_API_KEY")

        
        st.header("📁 Knowledge Base")

      
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload village Profile Documents",
            type=['pdf', 'txt', 'csv', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, CSV, or DOCX files containing village policies"
        )
        
        # Process uploaded files
        if uploaded_files and openai_api_key:
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing uploaded documents..."):
                    all_documents = []
                    st.session_state.uploaded_files = []  # Clear previous
                    
                    for uploaded_file in uploaded_files:
                        documents = process_uploaded_file(uploaded_file)
                        if documents:
                            all_documents.extend(documents)
                            st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    if all_documents:
                        # Create vector store
                        st.session_state.vector_store = create_vector_store(
                            all_documents, 
                            openai_api_key
                        )
                        
                        if st.session_state.vector_store:
                            # Create conversational chain
                            st.session_state.qa_chain = create_conversational_chain(
                                st.session_state.vector_store,
                                openai_api_key
                            )
                            st.session_state.vector_store_ready = True
                            st.success("✅ Documents processed successfully!")
                        else:
                            st.error("❌ Failed to create vector store")
                    else:
                        st.error("❌ No documents were processed successfully")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📄 Uploaded Files")
            for file_name in st.session_state.uploaded_files:
                st.write(f"• {file_name}")
        
        # Reset button
        if st.button("🗑️ Reset Everything"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.uploaded_files = []
            st.session_state.vector_store_ready = False
            st.rerun()
        
        # Status indicators
        st.header("📊 Status")
       
        
        if st.session_state.vector_store_ready:
            st.markdown('<p class="status-success">✅ Knowledge Base: Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Knowledge Base: Not Ready</p>', unsafe_allow_html=True)
        
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("Files Uploaded", len(st.session_state.uploaded_files))

         
        # Add export section
        create_export_section()

   

    # Main content area
    if not openai_api_key == os.getenv("OPENAI_API_KEY"):
        st.markdown("""
        <div class="upload-section">
            <h3>🔑 Getting Started</h3>
            <p>1. Enter your OpenAI API key in the sidebar</p>
            <p>2. Upload your village profile documents</p>
            <p>3. Click "Process Documents" to build the knowledge base</p>
            <p>4. Start asking questions about yourvillage profile!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.vector_store_ready:
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Knowledge Documents</h3>
            <p>Please upload your village profile documents in the sidebar to get started.</p>
            <p>Supported formats: PDF, TXT, CSV, DOCX</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display existing chat history
    display_chat_history()
 
    
    # Chat input
    if prompt := st.chat_input("Ask me about village profile..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_hr_policy(prompt, st.session_state.chat_history)
            
            st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update conversation history for context
        st.session_state.chat_history.append((prompt, response))

            
        # Rerun the app to refresh sidebar stats
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Note**: This chatbot uses OpenAI's GPT-4 model. Upload your village profile documents first to create a knowledge base for accurate responses.")
 
    

# Make sure these functions are defined BEFORE create_export_section()

def export_conversation_to_txt():
    """Export conversation history to text format"""
    if not st.session_state.messages:
        return None
    
    export_content = []
    export_content.append(" village profile Chatbot - Conversation Export")
    export_content.append("=" * 50)
    export_content.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    export_content.append(f"Total Messages: {len(st.session_state.messages)}")
    export_content.append(f"Uploaded Files: {', '.join(st.session_state.uploaded_files) if st.session_state.uploaded_files else 'None'}")
    export_content.append("=" * 50)
    export_content.append("")
    
    for i, message in enumerate(st.session_state.messages, 1):
        role = "User" if message["role"] == "user" else "Assistant"
        export_content.append(f"[{i}] {role}:")
        export_content.append(message["content"])
        export_content.append("-" * 30)
        export_content.append("")
    
    return "\n".join(export_content)

def export_conversation_to_json():
    """Export conversation history to JSON format"""
    if not st.session_state.messages:
        return None
    
    export_data = {
        "export_info": {
            "export_date": datetime.now().isoformat(),
            "total_messages": len(st.session_state.messages),
            "uploaded_files": st.session_state.uploaded_files,
            "chatbot_version": "village profiley Chatbot v2.0"
        },
        "conversation": st.session_state.messages
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def export_conversation_to_csv():
    """Export conversation history to CSV format"""
    if not st.session_state.messages:
        return None
    
    # Create DataFrame
    df_data = []
    for i, message in enumerate(st.session_state.messages, 1):
        df_data.append({
            "Message_ID": i,
            "Role": message["role"],
            "Content": message["content"],
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(df_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()



# NOW define create_export_section() AFTER the above functions
def create_export_section():
    """Create export section in the sidebar with multiple download options"""
    
    st.header("📤 Export Conversation")
    
    # Check if there are messages to export
    if not st.session_state.messages:
        st.info("No conversation to export yet. Start chatting to enable export!")
        return
    
    # Export statistics
    user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
    bot_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
    total_messages = len(st.session_state.messages)
    
    # Display basic stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👤 User Messages", user_messages)
    with col2:
        st.metric("🤖 Bot Messages", bot_messages)
    
    st.metric("📊 Total Messages", total_messages)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export Options
    st.subheader("📥 Download Options")
    
    # CSV Export
    csv_content = export_conversation_to_csv()
    if csv_content:
        st.download_button(
            label="📊 Download as CSV",
            data=csv_content,
            file_name=f"hr_chatbot_conversation_{timestamp}.csv",
            mime="text/csv",
            help="Download conversation as CSV file for spreadsheet analysis",
            use_container_width=True
        )
    
    # TXT Export
    txt_content = export_conversation_to_txt()
    if txt_content:
        st.download_button(
            label="📄 Download as TXT",
            data=txt_content,
            file_name=f"hr_chatbot_conversation_{timestamp}.txt",
            mime="text/plain",
            help="Download conversation as plain text file",
            use_container_width=True
        )
    
    # JSON Export
    json_content = export_conversation_to_json()
    if json_content:
        st.download_button(
            label="📋 Download as JSON",
            data=json_content,
            file_name=f"hr_chatbot_conversation_{timestamp}.json",
            mime="application/json",
            help="Download conversation as JSON file with metadata",
            use_container_width=True
        )
    
    # Preview Section
    st.subheader("👀 Preview")
    
    # Format selection for preview
    preview_format = st.selectbox(
        "Choose format to preview:",
        ["CSV", "TXT", "JSON"],
        help="Select which format you want to preview before downloading"
    )
    
    with st.expander(f"📄 {preview_format} Preview"):
        if preview_format == "CSV":
            if csv_content:
                try:
                    df = pd.read_csv(io.StringIO(csv_content))
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error previewing CSV: {str(e)}")
        
        elif preview_format == "TXT":
            if txt_content:
                st.text_area(
                    "Text Preview:",
                    value=txt_content[:1000] + "..." if len(txt_content) > 1000 else txt_content,
                    height=300,
                    disabled=True
                )
        
        elif preview_format == "JSON":
            if json_content:
                st.code(
                    json_content[:1000] + "..." if len(json_content) > 1000 else json_content,
                    language="json"
                )
    
    # Bulk Download Option
    st.subheader("📦 Bulk Download")
    
    if st.button("📤 Download All Formats", use_container_width=True):
        st.success("✅ All formats ready for download!")
        
        # Show all download buttons at once
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if csv_content:
                st.download_button(
                    label="📊 CSV",
                    data=csv_content,
                    file_name=f"hr_chatbot_conversation_{timestamp}.csv",
                    mime="text/csv",
                    key="bulk_csv"
                )
        
        with col2:
            if txt_content:
                st.download_button(
                    label="📄 TXT",
                    data=txt_content,
                    file_name=f"hr_chatbot_conversation_{timestamp}.txt",
                    mime="text/plain",
                    key="bulk_txt"
                )
        
        with col3:
            if json_content:
                st.download_button(
                    label="📋 JSON",
                    data=json_content,
                    file_name=f"hr_chatbot_conversation_{timestamp}.json",
                    mime="application/json",
                    key="bulk_json"
                )
    
    # Clear conversation option
    st.subheader("🗑️ Clear Data")
    
    if st.button("⚠️ Clear All Messages", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("✅ All messages cleared!")
        st.rerun()



if __name__ == "__main__":
    main()
