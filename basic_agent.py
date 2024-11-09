from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
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

# Define log filename with timestamp (using UTC)
LOG_FILENAME = f"agent_conversation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_UTC.log"

# Set up logging with UTC timestamp and custom formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Add newlines before and after each log entry
        return f"\n{datetime.utcnow().strftime('%d/%m/%Y UTC %H:%M:%S')} - {record.levelname} - {record.getMessage()}\n"

# Set up logging
handler = logging.FileHandler(LOG_FILENAME)
handler.setFormatter(CustomFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]  # File-only logging
)

# Get the OpenAI API key from the environment
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

os.environ["OPENAI_API_KEY"] = api_key

# Initialize tokenizer
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
except Exception as e:
    logging.error(f"Failed to initialize tokenizer: {str(e)}", exc_info=True)
    raise RuntimeError("Critical: Failed to initialize tokenizer")

def count_tokens(text):
    """Count the number of tokens in a string with error handling."""
    try:
        return len(tokenizer.encode(str(text)))  # Add str() to handle non-string inputs
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}", exc_info=True)
        return 0

# Define the path for the Chroma database
persist_directory = os.path.join(os.getcwd(), "chroma_db")

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
        
        # Verify database initialization
        count = vectordb._collection.count()
        logging.info(f"Successfully initialized Chroma database. Number of elements: {count}")
        return vectordb
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise RuntimeError("Critical: Failed to initialize vector database")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the AI agent with optional database clearing.")
parser.add_argument('--clear-db', action='store_true', help='Clear the existing database before initializing')
args = parser.parse_args()

# Initialize the database (clear only if --clear-db flag is set)
vectordb = initialize_database(clear=args.clear_db)

# Initialize the conversation memory with larger context
try:
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(temperature=0),
        max_token_limit=32000,  # 32K tokens
        return_messages=True,
        memory_key="history"
    )
    # Verify memory initialization
    memory.load_memory_variables({})  # Test load
    logging.info(f"Successfully initialized ConversationSummaryBufferMemory with max_token_limit=32000 (32K tokens)")
except Exception as e:
    logging.error(f"Failed to initialize memory: {str(e)}", exc_info=True)
    raise RuntimeError("Critical: Failed to initialize memory system")

def log_memory_stats():
    """Log detailed memory statistics with error handling."""
    try:
        memory_dict = memory.load_memory_variables({})
        current_messages = memory_dict.get("history", [])
        moving_summary = getattr(memory, 'moving_summary_buffer', "No summary yet")
        
        # Detailed token counting with safe handling
        messages_tokens = count_tokens(str(current_messages))
        summary_tokens = count_tokens(str(moving_summary))
        total_tokens = messages_tokens + summary_tokens
        
        logging.info("=== DETAILED MEMORY STATISTICS ===")
        logging.info(f"Memory Allocation (32K total):")
        logging.info(f"├── Current buffer: {messages_tokens:,} tokens ({(messages_tokens/32000*100):.1f}%)")
        logging.info(f"├── Summary buffer: {summary_tokens:,} tokens ({(summary_tokens/32000*100):.1f}%)")
        logging.info(f"└── Total usage: {total_tokens:,} tokens ({(total_tokens/32000*100):.1f}%)")
        logging.info(f"Remaining capacity: {32000-total_tokens:,} tokens ({((32000-total_tokens)/32000*100):.1f}%)")
        
        # Token usage warnings
        if total_tokens > 24000:  # 75% warning
            logging.warning("Token usage above 75% of capacity - consider implementing cleanup strategy")
        
        # Future multimedia reservation logging
        reserved_multimedia = 96000  # 96K tokens reserved for future multimedia
        logging.info("\nToken Reservation Status:")
        logging.info(f"├── Memory buffer (max): 32,000 tokens")
        logging.info(f"├── Reserved for multimedia: {reserved_multimedia:,} tokens")
        logging.info(f"└── Total context window: 128,000 tokens")
        
        # Detailed message analysis
        logging.info("\nMessage Buffer Analysis:")
        if isinstance(current_messages, list):
            for idx, msg in enumerate(current_messages, 1):
                msg_tokens = count_tokens(str(msg))
                logging.info(f"Message {idx}: {msg_tokens:,} tokens")
        
        logging.info("\nMoving Summary:")
        logging.info(f"{moving_summary}")
        logging.info("\nCurrent Messages:")
        logging.info(f"{current_messages}")
        logging.info("================================")
    except Exception as e:
        logging.error(f"Error in memory stats logging: {str(e)}", exc_info=True)

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant named John. The user's name is Ryan."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

def get_text_hash(text):
    """Generate a hash for the given text."""
    return hashlib.md5(text.encode()).hexdigest()

def interact_with_agent(user_input):
    """Interact with the agent with proper cleanup."""
    try:
        # Log pre-interaction stats
        logging.info("\n=== PRE-INTERACTION STATE ===")
        log_memory_stats()
        
        input_tokens = count_tokens(user_input)
        logging.info(f"\nNew Input Analysis:")
        logging.info(f"├── Input tokens: {input_tokens:,}")
        logging.info(f"└── Input text: {user_input}")

        # Retrieve relevant past conversations
        db_size = vectordb._collection.count()
        k = max(1, min(5, db_size))
        relevant_history = vectordb.similarity_search(user_input, k=k) if db_size > 0 else []
        
        logging.info(f"Retrieved {len(relevant_history)} relevant history items")
        for i, doc in enumerate(relevant_history):
            logging.info(f"History item {i + 1}:\n{doc.page_content}")
        
        # Combine current conversation with relevant history
        full_context = "Current user input:\n" + user_input
        if relevant_history:
            full_context = (
                "Relevant past information:\n" +
                "\n".join([doc.page_content for doc in relevant_history]) +
                "\n\n" + full_context
            )

        logging.info(f"Full context being sent to agent:\n{full_context}")

        conversation = ConversationChain(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
            prompt=prompt,
            memory=memory
        )

        # Get the response from the agent
        response = conversation.predict(input=full_context)
        
        # Log response analysis
        response_tokens = count_tokens(response)
        logging.info(f"\nResponse Analysis:")
        logging.info(f"├── Response tokens: {response_tokens:,}")
        logging.info(f"└── Response text: {response}")

        # Prepare new interaction for storage
        new_interaction = f"User: {user_input}\nAgent: {response}"
        interaction_hash = get_text_hash(new_interaction)

        # Get all existing entries
        existing_entries = vectordb.similarity_search(new_interaction, k=vectordb._collection.count()) if vectordb._collection.count() > 0 else []
        
        # Check for exact match
        is_duplicate = False
        for entry in existing_entries:
            if entry.page_content.strip() == new_interaction.strip():
                is_duplicate = True
                logging.info("=== EXACT DUPLICATE FOUND - NOT ADDING TO DATABASE ===")
                logging.info("Existing entry:")
                logging.info(f"{entry.page_content}")
                logging.info("New interaction (not stored):")
                logging.info(f"{new_interaction}")
                logging.info("============================================")
                break
        
        if not is_duplicate:
            # If no exact match found, add the new interaction
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(new_interaction)
            metadatas = [{"hash": interaction_hash} for _ in texts]
            
            # Add texts and verify storage
            vectordb.add_texts(texts, metadatas=metadatas)
            safe_persist_database()
            
            # Verify the entry was stored
            if not verify_database_entry(new_interaction):
                logging.error("Failed to verify database entry - attempting retry")
                # Retry once
                vectordb.add_texts(texts, metadatas=metadatas)
                safe_persist_database()
                
            logging.info("=== NEW UNIQUE INTERACTION - ADDING TO DATABASE ===")
            logging.info(f"Number of chunks added: {len(texts)}")
            logging.info("New interaction content:")
            logging.info(f"{new_interaction}")
            logging.info("============================================")

        # Update the conversation buffer with the response
        memory.save_context({"input": user_input}, {"output": response})
        
        logging.info("\n=== POST-INTERACTION STATE ===")
        log_memory_stats()

        # Check if summarization occurred
        if hasattr(memory, '_current_buffer_length'):
            if memory._current_buffer_length > memory.max_token_limit:
                logging.info("\n=== MEMORY SUMMARIZATION TRIGGERED ===")
                logging.info("├── Reason: Token limit exceeded")
                logging.info(f"├── Previous buffer length: {memory._current_buffer_length:,}")
                logging.info(f"└── New summary generated")

        return response
    except Exception as e:
        logging.error(f"Error in interaction: {str(e)}", exc_info=True)
        return "I'm sorry, but I encountered an error while processing your request."
    finally:
        try:
            safe_persist_database()
            logging.info(f"Database persisted. Elements: {vectordb._collection.count()}")
        except Exception as e:
            logging.error(f"Error in final persistence: {str(e)}", exc_info=True)

# Example interaction loop
while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            safe_persist_database()
            break
            
        response = interact_with_agent(user_input)
        print(f"\nAgent: {response}")
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logging.error(f"Main loop error: {str(e)}", exc_info=True)
        print("\nAn error occurred. Please try again.")
        safe_persist_database()  # Ensure persistence on error

# Final persistence after the loop
safe_persist_database()
logging.info(f"Final database state. Number of elements: {vectordb._collection.count()}")

def cleanup():
    """Enhanced cleanup with proper error handling."""
    logging.info("Starting cleanup process...")
    try:
        safe_persist_database()
        
        # Log final memory state
        logging.info("Final memory state:")
        log_memory_stats()
        
        # Close any open handlers
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
            
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")  # Use print as logging might be closed

def signal_handler(sig, frame):
    """Enhanced signal handler with proper cleanup."""
    print("\nGracefully shutting down...")
    cleanup()
    sys.exit(0)

# Update signal handling registration
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

if __name__ == "__main__":
    pass

def safe_persist_database():
    """Safely persist the database with error handling."""
    try:
        vectordb.persist()
        logging.info(f"Database successfully persisted. Elements: {vectordb._collection.count()}")
    except Exception as e:
        logging.error(f"Error persisting database: {str(e)}", exc_info=True)

def verify_database_entry(new_interaction, max_retries=3):
    """Verify database entry with retries."""
    for attempt in range(max_retries):
        try:
            results = vectordb.similarity_search(new_interaction, k=1)
            if results and results[0].page_content.strip() == new_interaction.strip():
                logging.info(f"Database entry verification: SUCCESS (attempt {attempt + 1})")
                return True
            logging.warning(f"Verification attempt {attempt + 1} failed")
            time.sleep(0.5)  # Short delay between retries
        except Exception as e:
            logging.error(f"Database verification error (attempt {attempt + 1}): {str(e)}", exc_info=True)
    
    logging.error("All verification attempts failed")
    return False
