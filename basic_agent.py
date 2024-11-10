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

try:
    tokenizer = tiktoken.encoding_for_model(MODEL_SELECTION)
except Exception as e:
    logging.error(f"Failed to initialize tokenizer: {str(e)}", exc_info=True)
    raise RuntimeError("Critical: Failed to initialize tokenizer")

# Define the path for the Chroma database
persist_directory = os.path.join(os.getcwd(), "chroma_db")

def get_text_hash(text):
    """Generate a hash for the given text."""
    return hashlib.md5(text.encode()).hexdigest()

def count_tokens(text):
    """Count the number of tokens in a string with error handling."""
    try:
        return len(tokenizer.encode(str(text)))
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}", exc_info=True)
        return 0

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
    """Log detailed memory statistics with error handling."""
    try:
        memory_dict = memory.load_memory_variables({})
        current_messages = memory_dict.get("history", [])
        moving_summary = getattr(memory, 'moving_summary_buffer', "No summary yet")
        
        messages_tokens = count_tokens(str(current_messages))
        summary_tokens = count_tokens(str(moving_summary))
        total_tokens = messages_tokens + summary_tokens
        
        # Memory summarization threshold for testing
        if total_tokens > 4000:  # About 12.5% of 32K
            logging.warning("MEMORY SUMMARIZATION TRIGGERED - Token usage exceeds threshold")
        
        logging.info("=== DETAILED MEMORY STATISTICS ===")
        logging.info(f"Memory Allocation (32K total):")
        logging.info(f"├── Current buffer: {messages_tokens:,} tokens ({(messages_tokens/MAX_CONVERSATION_TOKENS*100):.1f}%)")
        logging.info(f"├── Summary buffer: {summary_tokens:,} tokens ({(summary_tokens/MAX_CONVERSATION_TOKENS*100):.1f}%)")
        logging.info(f"└── Total usage: {total_tokens:,} tokens ({(total_tokens/MAX_CONVERSATION_TOKENS*100):.1f}%)")
        
        logging.info("\nMessage Buffer Analysis:")
        if isinstance(current_messages, list):
            for idx, msg in enumerate(current_messages, 1):
                msg_tokens = count_tokens(str(msg))
                logging.info(f"Message {idx}: {msg_tokens:,} tokens")
        
        logging.info(f"\nMoving Summary ({summary_tokens} tokens):")
        logging.info(f"{moving_summary}\n")
        
    except Exception as e:
        logging.error(f"Error logging memory stats: {str(e)}", exc_info=True)

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

# Initialize memory
try:
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
        max_token_limit=MAX_CONVERSATION_TOKENS,
        return_messages=True,
        memory_key="history"
    )
    memory.load_memory_variables({})  # Test load
    logging.info("Successfully initialized ConversationSummaryBufferMemory")
except Exception as e:
    logging.error(f"Failed to initialize memory: {str(e)}", exc_info=True)
    raise RuntimeError("Critical: Failed to initialize memory system")

# Initialize database
vectordb = initialize_database(clear=False)

def interact_with_agent(user_input):
    """Interact with the agent with proper cleanup and temporal awareness."""
    try:
        # Log pre-interaction stats
        logging.info("\n=== PRE-INTERACTION STATE ===")
        log_memory_stats()
        
        # Get current timestamp at the moment of interaction
        current_timestamp = datetime.utcnow()
        timestamped_input = f"[{current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')}]\nUser: {user_input}"
        
        # Retrieve relevant past conversations
        db_size = vectordb._collection.count()
        k = max(1, min(5, db_size))
        relevant_history = vectordb.similarity_search(user_input, k=k) if db_size > 0 else []
        
        # Combine current conversation with relevant history
        full_context = f"Current interaction:\n{timestamped_input}"
        if relevant_history:
            full_context = (
                "Relevant past information:\n" +
                "\n".join([doc.page_content for doc in relevant_history]) +
                "\n\n" + full_context
            )

        # Get response from agent
        logging.info("\n=== FULL CONTEXT FOR AGENT ===")
        logging.info(f"Number of relevant history items: {len(relevant_history)}")
        logging.info(f"Full context being sent to agent:\n{full_context}")
        logging.info("================================")

        conversation = ConversationChain(
            llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
            prompt=ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant with perfect memory recall."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ]),
            memory=memory
        )
        response = conversation.predict(input=full_context)
        
        # Create the complete interaction record
        new_interaction = (
            f"[{current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')}]\n"
            f"User: {user_input}\n"
            f"Agent: {response}"
        )
        interaction_hash = get_text_hash(new_interaction)

        # Check for duplicates and store
        existing_entries = vectordb.similarity_search(new_interaction, k=1)
        is_duplicate = any(entry.page_content.strip() == new_interaction.strip() 
                         for entry in existing_entries)

        if not is_duplicate:
            logging.info("\n=== STORING NEW UNIQUE INTERACTION ===")
            logging.info(f"Interaction Hash: {interaction_hash}")
            logging.info(f"New Interaction:\n{new_interaction}")
            logging.info("====================================")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(new_interaction)
            metadatas = [{
                "hash": interaction_hash,
                "timestamp": current_timestamp.isoformat(),
                "timestamp_readable": current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')
            } for _ in texts]
            
            vectordb.add_texts(texts, metadatas=metadatas)
            
            if not verify_database_entry(new_interaction):
                logging.error("Failed to verify database entry - attempting retry")
                vectordb.add_texts(texts, metadatas=metadatas)

        # Update conversation memory
        memory.save_context({"input": user_input}, {"output": response})
        
        # Log post-interaction stats
        logging.info("\n=== POST-INTERACTION STATE ===")
        log_memory_stats()

        return response
        
    except Exception as e:
        logging.error(f"Error in interaction: {str(e)}", exc_info=True)
        return "I'm sorry, but I encountered an error while processing your request."

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
