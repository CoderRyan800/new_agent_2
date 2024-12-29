from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tiktoken
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
import shutil
import argparse
import hashlib
import signal
import sys
import atexit
import logging
from datetime import datetime
import time
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Constants
MAX_CONVERSATION_TOKENS = 8000
MODEL_SELECTION = "gpt-4o"

# Define log filename with timestamp (using UTC)
LOG_FILENAME = f"agent_conversation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_UTC.log"

# Set up logging with UTC timestamp and custom formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"[{datetime.utcnow().strftime('%d/%m/%Y UTC %H:%M:%S')}] - {record.levelname} - {record.getMessage()}"

# Set up logging
handler = logging.FileHandler(LOG_FILENAME)
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Initialize core components
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model(MODEL_SELECTION)

# Define the path for the Chroma database
persist_directory = os.path.join(os.getcwd(), "chroma_db")

def get_text_hash(text):
    """Generate a hash for the given text."""
    return hashlib.md5(text.encode()).hexdigest()

def count_tokens(text):
    """Simple token counter."""
    return len(tokenizer.encode(str(text)))

def safe_persist_database():
    """Check database state with error handling."""
    try:
        # Note: Persistence is now automatic in Chroma 0.4.x
        count = vectordb._collection.count()
        logging.info(f"Database state checked. Elements: {count}")
    except Exception as e:
        logging.error(f"Error checking database: {str(e)}", exc_info=True)

def initialize_database(clear=False):
    """Initialize the vector database with proper error handling."""
    try:
        if clear and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            logging.info(f"Cleared existing database at {persist_directory}")
        
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        count = vectordb._collection.count()
        logging.info(f"Successfully initialized Chroma database. Elements: {count}")
        return vectordb
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise RuntimeError("Critical: Failed to initialize vector database")

def verify_database_entry(new_interaction, max_retries=3):
    """Verify database entry with retries."""
    for attempt in range(max_retries):
        try:
            results = vectordb.similarity_search(new_interaction, k=1)
            if results and results[0].page_content.strip() == new_interaction.strip():
                logging.info(f"Database entry verification: SUCCESS (attempt {attempt + 1})")
                return True
            logging.warning(f"Verification attempt {attempt + 1} failed")
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Database verification error (attempt {attempt + 1}): {str(e)}", exc_info=True)
    
    logging.error("All verification attempts failed")
    return False

def log_memory_stats():
    """Simplified memory statistics logging."""
    try:
        memory_dict = memory.load_memory_variables({})
        current_messages = memory_dict.get("history", [])
        
        total_tokens = count_tokens(str(current_messages))
        logging.info(f"Memory usage: {total_tokens:,} tokens ({(total_tokens/MAX_CONVERSATION_TOKENS*100):.1f}% of limit)")
        
    except Exception as e:
        logging.error(f"Error logging memory stats: {str(e)}")

def cleanup():
    """Cleanup function for graceful shutdown."""
    try:
        safe_persist_database()
        logging.info("Cleanup completed successfully")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def signal_handler(sig, frame):
    """Enhanced signal handler with proper cleanup."""
    print("\nGracefully shutting down...")
    cleanup()
    sys.exit(0)

class InMemoryHistory(BaseChatMessageHistory):
    """Simple in-memory message store."""
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    @property
    def messages(self):
        return self.messages

def get_session_history() -> BaseChatMessageHistory:
    """Returns a new chat message history instance."""
    return ChatMessageHistory()

# Initialize memory with token limit
memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
    max_token_limit=MAX_CONVERSATION_TOKENS,
    return_messages=True,
    memory_key="history"
)

# Initialize database
vectordb = initialize_database(clear=False)

# Replace the ConversationChain with RunnableWithMessageHistory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with perfect memory recall."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | ChatOpenAI(model_name=MODEL_SELECTION, temperature=0)
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def interact_with_agent(user_input):
    """Core interaction function with basic token monitoring."""
    try:
        log_memory_stats()
        
        current_timestamp = datetime.utcnow()
        timestamp_str = current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')
        
        # Get relevant history
        relevant_history = vectordb.similarity_search(user_input, k=3) if vectordb._collection.count() > 0 else []
        
        # Prepare context
        context = f"[{timestamp_str}]\nUser: {user_input}"
        if relevant_history:
            context = "Past context:\n" + "\n".join(doc.page_content for doc in relevant_history) + "\n\n" + context
        
        # Get response using the new pattern
        response = conversation.invoke(
            {"input": context},
            config={"configurable": {"session_id": "default"}}
        )
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Store interaction
        new_interaction = f"[{timestamp_str}]\nUser: {user_input}\nAgent: {response_text}"
        vectordb.add_texts(
            texts=[new_interaction],
            metadatas=[{
                "timestamp": current_timestamp.isoformat(),
                "timestamp_readable": timestamp_str
            }]
        )
        
        log_memory_stats()
        
        return response_text
        
    except Exception as e:
        logging.error(f"Interaction error: {str(e)}")
        return "I encountered an error processing your request."

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

# Only run interactive loop if script is run directly
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                cleanup()
                break
                
            response = interact_with_agent(user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
            signal_handler(signal.SIGINT, None)
        except Exception as e:
            logging.error(f"Main loop error: {str(e)}", exc_info=True)
            print("\nAn error occurred. Please try again.")
