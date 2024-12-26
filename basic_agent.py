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
import shutil
import argparse
import hashlib
import signal
import sys
import atexit
import logging
from datetime import datetime
import time
from langchain.agents import AgentType, initialize_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Constants
MAX_CONVERSATION_TOKENS = 32000
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

# Define the path for the Chroma database
persist_directory = os.path.join(os.getcwd(), "chroma_db")

# Add this before the AgentManager class
def signal_handler(sig, frame):
    """Enhanced signal handler with proper cleanup."""
    print("\nGracefully shutting down...")
    if 'agent_manager' in globals():
        agent_manager.cleanup()
    sys.exit(0)

# Set up signal handlers at module level
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class AgentManager:
    def __init__(self, model_name="gpt-4o", clear_db=False, persist_dir=None):
        self.model_name = model_name
        self.max_tokens = MAX_CONVERSATION_TOKENS
        self.persist_directory = persist_dir or os.path.join(os.getcwd(), "chroma_db")
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer: {str(e)}", exc_info=True)
            raise RuntimeError("Critical: Failed to initialize tokenizer")
            
        # Initialize components
        self.vectordb = self.initialize_database(clear=clear_db)
        self.memory = self.create_memory()
        self.agent = self.create_agent()
        
    def create_memory(self):
        return ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model_name=self.model_name, temperature=0),
            max_token_limit=self.max_tokens,
            memory_key="chat_history",
            return_messages=True
        )
    
    def create_agent(self):
        llm = ChatOpenAI(model_name=self.model_name)
        tools = [
            Tool(
                name="Search Conversation History",
                func=lambda q: self.vectordb.similarity_search(q),
                description="Search previous conversations"
            )
        ]
        
        # Create a custom prompt template with all required variables
        system_message = """You are a helpful AI assistant. You have access to the following tools:

{tools}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def interact(self, user_input: str) -> str:
        """Interact with the agent with proper cleanup and temporal awareness."""
        try:
            # Log pre-interaction stats
            logging.info("\n=== PRE-INTERACTION STATE ===")
            self.log_memory_stats()
            
            # Get current timestamp at the moment of interaction
            current_timestamp = datetime.utcnow()
            timestamped_input = f"[{current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')}]\nUser: {user_input}"
            
            # Retrieve relevant past conversations
            db_size = self.vectordb._collection.count()
            k = max(1, min(5, db_size))
            relevant_history = self.vectordb.similarity_search(user_input, k=k) if db_size > 0 else []
            
            # Build context with temporal information
            full_context = f"Current interaction:\n{timestamped_input}"
            if relevant_history:
                full_context = (
                    "Relevant past information:\n" +
                    "\n".join([doc.page_content for doc in relevant_history]) +
                    "\n\n" + full_context
                )

            # Get response from agent using the new API
            response = self.agent.invoke({"input": full_context})["output"]
            
            # Create the complete interaction record with timestamp
            new_interaction = (
                f"[{current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')}]\n"
                f"User: {user_input}\n"
                f"Agent: {response}"
            )
            
            # Store in vector database with metadata
            interaction_hash = self.get_text_hash(new_interaction)
            self.store_interaction(new_interaction, current_timestamp, interaction_hash)
            
            # Log post-interaction stats
            logging.info("\n=== POST-INTERACTION STATE ===")
            self.log_memory_stats()
            
            return response
            
        except Exception as e:
            logging.error(f"Error in interaction: {str(e)}", exc_info=True)
            return "I'm sorry, but I encountered an error while processing your request."

    def store_interaction(self, new_interaction, timestamp, interaction_hash):
        """Store interaction in vector database with verification."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(new_interaction)
            metadatas = [{
                "hash": interaction_hash,
                "timestamp": timestamp.isoformat(),
                "timestamp_readable": timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')
            } for _ in texts]
            
            self.vectordb.add_texts(texts, metadatas=metadatas)
            
            if not self.verify_database_entry(new_interaction):
                logging.error("Failed to verify database entry - attempting retry")
                self.vectordb.add_texts(texts, metadatas=metadatas)
                
        except Exception as e:
            logging.error(f"Error storing interaction: {str(e)}", exc_info=True)

    def log_memory_stats(self):
        """Log detailed memory statistics with error handling."""
        try:
            memory_dict = self.memory.load_memory_variables({})
            current_messages = memory_dict.get("chat_history", [])
            moving_summary = getattr(self.memory, 'moving_summary_buffer', "No summary yet")
            
            messages_tokens = self.count_tokens(str(current_messages))
            summary_tokens = self.count_tokens(str(moving_summary))
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
                    msg_tokens = self.count_tokens(str(msg))
                    logging.info(f"Message {idx}: {msg_tokens:,} tokens")
            
            logging.info(f"\nMoving Summary ({summary_tokens} tokens):")
            logging.info(f"{moving_summary}\n")
            
        except Exception as e:
            logging.error(f"Error logging memory stats: {str(e)}", exc_info=True)

    def count_tokens(self, text):
        """Count the number of tokens in a string with error handling."""
        try:
            return len(self.tokenizer.encode(str(text)))
        except Exception as e:
            logging.error(f"Error counting tokens: {str(e)}", exc_info=True)
            return 0

    def get_text_hash(self, text):
        """Generate a hash for the given text."""
        return hashlib.md5(text.encode()).hexdigest()

    def verify_database_entry(self, new_interaction, max_retries=3):
        """Verify database entry with retries."""
        for attempt in range(max_retries):
            try:
                results = self.vectordb.similarity_search(new_interaction, k=1)
                if results and results[0].page_content.strip() == new_interaction.strip():
                    logging.info(f"Database entry verification: SUCCESS (attempt {attempt + 1})")
                    return True
                logging.warning(f"Verification attempt {attempt + 1} failed")
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Database verification error (attempt {attempt + 1}): {str(e)}", exc_info=True)
        
        logging.error("All verification attempts failed")
        return False

    def cleanup(self):
        """Cleanup function for graceful shutdown."""
        try:
            # Note: Persistence is now automatic in Chroma 0.4.x
            count = self.vectordb._collection.count()
            logging.info(f"Database state checked. Elements: {count}")
            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def initialize_database(self, clear=False):
        """Initialize the vector database with proper error handling."""
        try:
            if clear and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logging.info(f"Cleared existing database at {self.persist_directory}")
            
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            count = vectordb._collection.count()
            logging.info(f"Successfully initialized Chroma database. Elements: {count}")
            return vectordb
        except Exception as e:
            logging.error(f"Failed to initialize database: {str(e)}", exc_info=True)
            raise RuntimeError("Critical: Failed to initialize vector database")

# Usage would then be:
if __name__ == "__main__":
    agent_manager = AgentManager()
    
    def cleanup_handler():
        """Cleanup handler for atexit."""
        print("\nPerforming cleanup...")
        agent_manager.cleanup()
    
    atexit.register(cleanup_handler)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                agent_manager.cleanup()
                break
            
            response = agent_manager.interact(user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
            signal_handler(signal.SIGINT, None)
        except Exception as e:
            logging.error(f"Main loop error: {str(e)}", exc_info=True)
            print("\nAn error occurred. Please try again.")
