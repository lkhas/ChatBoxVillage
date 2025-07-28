"""
HR Policy Chatbot - User Interface with RAG Implementation
This module provides a chatbot interface for HR policy queries using 
Retrieval Augmented Generation (RAG) with LangChain and OpenAI.
"""

import os
import streamlit as st
from typing import List, Optional, Dict, Any
import datetime
import json

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Import the knowledge base manager
from village_profile_upload import HRPolicyKnowledgeBase

class HRPolicyChatbot:
    """
    HR Policy Chatbot with RAG capabilities.
    """
    
    def __init__(self, openai_api_key: str, vector_store_path: str = "hr_policy_vectorstore"):
        """
        Initialize the HR Policy Chatbot.
        
        Args:
            openai_api_key: OpenAI API key
            vector_store_path: Path to the vector store
        """
        self.openai_api_key = openai_api_key
        self.vector_store_path = vector_store_path
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="GPT-3.5",
            temperature=0.3
        )
        
        # Initialize knowledge base
        self.knowledge_base = HRPolicyKnowledgeBase(openai_api_key, vector_store_path)
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize RAG chain
        self.rag_chain = None
        self.setup_rag_chain()
        
        # Custom prompt template
        self.hr_prompt_template = """
        You are an HR Policy Assistant chatbot. Your role is to help employees understand company HR policies and procedures.
        
        Use the following context from the company's HR policy documents to answer the human's question.
        If you don't know the answer based on the provided context, say "I don't have information about that specific policy in our knowledge base. Please consult with HR directly or refer to the complete policy documents."
        
        Always be professional, helpful, and accurate. If the question involves sensitive HR matters, remind the user to also consult with HR personnel directly.
        
        Context from HR Policy Documents:
        {context}
        
        Previous conversation:
        {chat_history}
        
        Human: {question}
        
        HR Assistant: """
    
    def setup_rag_chain(self):
        """Set up the RAG chain for question answering."""
        try:
            if self.knowledge_base.vectorstore is None:
                st.warning("No knowledge base loaded. Please upload HR policy documents first.")
                return
            
            # Create custom prompt
            prompt = PromptTemplate(
                template=self.hr_prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            # Create the conversational retrieval chain
            self.rag_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.knowledge_base.vectorstore.as_retriever(search_kwargs={"k": 4}),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
            
        except Exception as e:
            st.error(f"Error setting up RAG chain: {str(e)}")
    
    def get_response(self, user_question: str) -> Dict[str, Any]:
        """
        Get response from the HR Policy Chatbot.
        
        Args:
            user_question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.rag_chain:
            return {
                "answer": "I'm sorry, but the HR policy knowledge base is not available. Please upload HR policy documents first.",
                "source_documents": []
            }
        
        try:
            # Get response from RAG chain
            response = self.rag_chain({"question": user_question})
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", [])
            }
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "source_documents": []
            }
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        st.success("Conversation history cleared!")
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history."""
        return self.memory.chat_memory.messages
    
    def export_conversation(self) -> str:
        """Export conversation history as JSON."""
        messages = self.get_conversation_history()
        conversation_data = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_data.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation_data.append({"role": "assistant", "content": msg.content})
        
        return json.dumps(conversation_data, indent=2)


def main():
    """Main function to run the HR Policy Chatbot interface."""
    st.set_page_config(
        page_title="HR Policy Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #234c6b;
        justify-content: flex-end;
    }
    .bot-message {
        background-color: #544056;
        justify-content: flex-start;
    }
    .message-content {
        max-width: 80%;
        padding: 0.5rem;
    }
    .source-docs {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 HR Policy Chatbot")
    st.markdown("Ask me anything about company HR policies and procedures!")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # OpenAI API Key input
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = HRPolicyChatbot(openai_api_key)
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Knowledge base stats
    if st.session_state.chatbot.knowledge_base:
        stats = st.session_state.chatbot.knowledge_base.get_knowledge_base_stats()
        st.sidebar.metric("Knowledge Base Documents", stats["total_documents"])
        st.sidebar.info(stats["status"])
    
    # Clear conversation button
    if st.sidebar.button("Clear Conversation"):
        st.session_state.chatbot.clear_conversation_history()
        if "messages" in st.session_state:
            del st.session_state.messages
        st.rerun()
    
    # Export conversation
    if st.sidebar.button("Export Conversation"):
        conversation_json = st.session_state.chatbot.export_conversation()
        st.sidebar.download_button(
            label="Download Conversation",
            data=conversation_json,
            file_name=f"hr_chat_conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Main chat interface
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """
        Hello! I'm your HR Policy Assistant. I can help you with questions about:
        
        • Company policies and procedures
        • Employee benefits and compensation
        • Leave policies and time off
        • Performance management
        • Code of conduct and ethics
        • And other HR-related topics
        
        What would you like to know about our HR policies?
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display source documents if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Source Documents"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source.metadata.get('source_file', 'Unknown')}")
                        st.markdown(f"```\n{source.page_content[:300]}...\n```")
    
    # Chat input
    if prompt := st.chat_input("Ask me about HR policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(prompt)
                
                # Display response
                st.markdown(response["answer"])
                
                # Display source documents
                if response["source_documents"]:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.markdown(f"**Source {i}:** {doc.metadata.get('source_file', 'Unknown')}")
                            st.markdown(f"```\n{doc.page_content[:300]}...\n```")
        
        # Add bot response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["answer"],
            "sources": response["source_documents"]
        })
    
    # Help section
    with st.sidebar:
        st.header("Help & Tips")
        
        with st.expander("❓ How to use"):
            st.markdown("""
            1. **Upload Documents**: First, use the upload tool to add HR policy documents
            2. **Ask Questions**: Type your HR-related questions in the chat
            3. **Review Sources**: Check the source documents for detailed information
            4. **Clear Chat**: Use the clear button to start a new conversation
            """)
        
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - "What is the vacation policy?"
            - "How do I request time off?"
            - "What are the company benefits?"
            - "What is the dress code policy?"
            - "How does the performance review process work?"
            - "What is the policy on remote work?"
            """)
        
        with st.expander("⚠️ Important Notes"):
            st.markdown("""
            - This chatbot provides information based on uploaded HR documents
            - For sensitive or personal HR matters, always consult with HR directly
            - The responses are AI-generated and should be verified with official policies
            - Keep your conversations professional and appropriate
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This chatbot provides information based on uploaded HR policy documents. "
        "For official policy interpretations or sensitive matters, please consult with HR personnel directly."
    )


if __name__ == "__main__":
    main()