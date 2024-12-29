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
PERSIST_DIRECTORY = "chroma_db"

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

class Agent:
    def __init__(self):
        # Initialize core components
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize components
        self.tokenizer = tiktoken.encoding_for_model(MODEL_SELECTION)
        self.vectordb = self.initialize_database(clear=False)
        self.memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
            max_token_limit=MAX_CONVERSATION_TOKENS,
            return_messages=True,
            memory_key="history"
        )

        # Initialize conversation chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with perfect memory recall."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | ChatOpenAI(model_name=MODEL_SELECTION, temperature=0)
        self.conversation = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    # Move all standalone functions into class methods
    def get_text_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def count_tokens(self, text):
        return len(self.tokenizer.encode(str(text)))

    def safe_persist_database(self):
        try:
            count = self.vectordb._collection.count()
            logging.info(f"Database state checked. Elements: {count}")
        except Exception as e:
            logging.error(f"Error checking database: {str(e)}", exc_info=True)

    def initialize_database(self, clear=False):
        """Initialize the vector database with proper error handling."""
        try:
            if clear and os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY)
                logging.info(f"Cleared existing database at {PERSIST_DIRECTORY}")
            
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            count = vectordb._collection.count()
            logging.info(f"Successfully initialized Chroma database. Elements: {count}")
            return vectordb
        except Exception as e:
            logging.error(f"Failed to initialize database: {str(e)}", exc_info=True)
            raise RuntimeError("Critical: Failed to initialize vector database")

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

    def log_memory_stats(self):
        """Simplified memory statistics logging."""
        try:
            memory_dict = self.memory.load_memory_variables({})
            current_messages = memory_dict.get("history", [])
            
            total_tokens = self.count_tokens(str(current_messages))
            logging.info(f"Memory usage: {total_tokens:,} tokens ({(total_tokens/MAX_CONVERSATION_TOKENS*100):.1f}% of limit)")
            
        except Exception as e:
            logging.error(f"Error logging memory stats: {str(e)}")

    def cleanup(self):
        """Cleanup function for graceful shutdown."""
        try:
            self.safe_persist_database()
            logging.info("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def signal_handler(self, sig, frame):
        """Enhanced signal handler with proper cleanup."""
        print("\nGracefully shutting down...")
        self.cleanup()
        sys.exit(0)

    def get_session_history(self) -> BaseChatMessageHistory:
        """Returns a new chat message history instance."""
        return ChatMessageHistory()

    def interact_with_agent(self, user_input):
        """Core interaction function with basic token monitoring."""
        try:
            self.log_memory_stats()
            
            current_timestamp = datetime.utcnow()
            timestamp_str = current_timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')
            
            # Get relevant history
            relevant_history = self.vectordb.similarity_search(user_input, k=3) if self.vectordb._collection.count() > 0 else []
            
            # Prepare context
            context = f"[{timestamp_str}]\nUser: {user_input}"
            if relevant_history:
                context = "Past context:\n" + "\n".join(doc.page_content for doc in relevant_history) + "\n\n" + context
            
            # Get response using the new pattern
            response = self.conversation.invoke(
                {"input": context},
                config={"configurable": {"session_id": "default"}}
            )
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Store interaction
            new_interaction = f"[{timestamp_str}]\nUser: {user_input}\nAgent: {response_text}"
            self.vectordb.add_texts(
                texts=[new_interaction],
                metadatas=[{
                    "timestamp": current_timestamp.isoformat(),
                    "timestamp_readable": timestamp_str
                }]
            )
            
            self.log_memory_stats()
            
            return response_text
            
        except Exception as e:
            logging.error(f"Interaction error: {str(e)}")
            return "I encountered an error processing your request."

    def run(self):
        """Interactive loop to run the agent."""
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!")
                    self.cleanup()
                    break
                    
                response = self.interact_with_agent(user_input)
                print(f"\nAgent: {response}")
                
            except KeyboardInterrupt:
                print("\nReceived interrupt signal...")
                self.cleanup()
                sys.exit(0)
            except Exception as e:
                logging.error(f"Main loop error: {str(e)}", exc_info=True)
                print("\nAn error occurred. Please try again.")

# Update the main block
if __name__ == "__main__":
    agent = Agent()
    agent.run()
