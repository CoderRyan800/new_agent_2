from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import tiktoken
import shutil
import logging
import time
from datetime import datetime
from pathlib import Path
import sys
import json
import signal

# Import our utilities
from ..utils.memory_utils import MemoryUtils

# Path Configuration and Validation
def validate_project_structure():
    """Validate and setup project directory structure."""
    base_path = Path(__file__).parent.parent.parent.resolve()
    
    # Debug output
    print(f"Resolved paths:")
    print(f"BASE_PATH: {base_path}")
    
    # Verify project root
    if not base_path.name == 'new_agent_2':
        raise RuntimeError(f"Incorrect project root: {base_path}. Expected 'new_agent_2'")
    
    # Define critical paths
    paths = {
        'data': base_path / 'data',
        'vector_stores': base_path / 'data' / 'vector_stores',
        'automatic': base_path / 'data' / 'vector_stores' / 'automatic',
        'manual': base_path / 'data' / 'vector_stores' / 'manual',
        'context_files': base_path / 'data' / 'context_files',
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Verified/created directory: {path}")
    
    return base_path, paths

# Get validated paths
BASE_PATH, PATHS = validate_project_structure()

# Constants
MAX_CONVERSATION_TOKENS = 8000
MODEL_SELECTION = "gpt-4"
PERSIST_DIRECTORY = str(PATHS['automatic'])
SECOND_VECTOR_DB = str(PATHS['manual'])
CONTEXT_FILE = str(PATHS['context_files'] / 'context_memory.json')

# Ensure directories exist
os.makedirs(os.path.dirname(CONTEXT_FILE), exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(SECOND_VECTOR_DB, exist_ok=True)

# Should add path existence checks
if not os.path.exists(BASE_PATH):
    raise RuntimeError(f"Base path not found: {BASE_PATH}")

class Agent:
    def __init__(self):
        """Initialize the agent with all necessary components."""
        # Set up signal handlers first
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Verify API key
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize core components
        self.tokenizer = tiktoken.encoding_for_model(MODEL_SELECTION)
        self.vectordb = self.initialize_database(clear=False)
        self.memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
            max_token_limit=MAX_CONVERSATION_TOKENS,
            return_messages=True,
            memory_key="history"
        )

        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Create agent components
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()

        logging.info("Agent initialized successfully")

    def _initialize_tools(self):
        """Initialize the agent's tools."""
        return [
            Tool(
                name="Calculator",
                func=MemoryUtils.safe_calculator,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression using numbers and basic operators (+, -, *, /)."
            ),
            Tool(
                name="EditContext",
                func=lambda x: MemoryUtils.edit_context(x, CONTEXT_FILE),
                description="Edit the context file. Use format 'header|content' where header must be one of: persona, human, context_notes. Example: 'persona|I am a helpful AI assistant'. Use this to update your persona, your knowledge about the human, or your working notes. Keep context_notes brief and consider moving longer content to manual memory."
            ),
            Tool(
                name="InitializeManualMemory",
                func=lambda: MemoryUtils.initialize_manual_memory(SECOND_VECTOR_DB),
                description="Initialize or check your long-term manual memory storage system. Run this at startup to ensure your memory system is ready. This is like preparing your permanent notebook."
            ),
            Tool(
                name="WriteToMemory",
                func=lambda x: MemoryUtils.write_to_manual_memory(x, SECOND_VECTOR_DB),
                description="Store important information in your long-term manual memory. Use format 'title|content'. Include keywords and context in your content to make future retrieval easier. Use this when information is too detailed for context_notes or when you want to preserve something for future reference. Think of this as writing in your permanent notebook with good indexing."
            ),
            Tool(
                name="SearchMemory",
                func=lambda x: MemoryUtils.search_manual_memory(x, SECOND_VECTOR_DB),
                description="Search through your long-term manual memory storage. Input any search query to find relevant stored memories. Use this to recall detailed information you've stored. This searches your permanent notebook for relevant entries."
            )
        ]

    def _create_agent(self):
        """Create the agent with the ReAct prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with perfect memory recall and access to tools. At the start of each interaction, you will see three important context sections:

1. "persona": This defines who you are and how you should behave. You should always act according to this persona.
2. "human": This describes the human you're working with. Use this to better understand and assist your user.
3. "context_notes": This is your notepad for important facts or observations. Keep these notes focused and brief. When they grow too long or contain detailed information you might need later, move appropriate content to your manual memory storage using the WriteToMemory tool.

You have two memory systems:
1. Context Notes: Your quick-access notepad for current, brief information
2. Manual Memory: Your permanent notebook for detailed information, important facts, and anything you might need to recall later

Memory Management Strategy:
- Keep context_notes brief and current
- When context_notes grow long, consider what to move to manual memory
- Use WriteToMemory to store important information with good keywords and context
- Use SearchMemory to recall stored information when needed
- Always include clear titles and detailed context when storing memories to make future retrieval easier

You have access to the following tools:

{tools}

The available tools are: {tool_names}

You must ALWAYS respond using the following format, even for simple greetings or responses that don't require tools:

Thought: First, I should think about what to do
Thought: I now know how to respond
Final Answer: [Your actual response here]

If you need to use a tool, use this format instead:
Thought: First, I should...
Action: tool_name
Action Input: input for the tool
Observation: tool output
... (repeat Thought/Action/Observation if needed)
Thought: I now know the answer
Final Answer: [Your response here]"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        return create_react_agent(
            llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
            tools=self.tools,
            prompt=prompt
        )

    def _create_agent_executor(self):
        """Create the agent executor with proper configuration."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

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
            
            # Get context and history
            context_file_content = self.read_context_file()
            relevant_history = self.vectordb.similarity_search(user_input, k=3) if self.vectordb._collection.count() > 0 else []
            
            # Build context
            context = context_file_content
            if relevant_history:
                context += "\n\nPast context:\n" + "\n".join(doc.page_content for doc in relevant_history)
            context += f"\n\n[{timestamp_str}]\nUser: {user_input}"
            
            logging.info(f"Context: {context}")
            
            # Get response from agent
            response = self.agent_executor.invoke({"input": context})
            response_text = response.get('output', str(response))
            
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
            logging.error(f"Interaction error: {str(e)}", exc_info=True)
            return "I encountered an error processing your request."

    def run(self):
        """Interactive loop to run the agent."""
        print("\nAgent initialized and ready. Type 'exit', 'quit', or 'bye' to end the session.")
        print("Type your message and press Enter to begin.\n")
        
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

    def read_context_file(self):
        """
        Reads the context file and formats it into a string with headers and values.
        Returns formatted string or empty string if file doesn't exist/has error.
        """
        try:
            with open(CONTEXT_FILE, 'r') as file:
                context_data = json.loads(file.read())
            
            formatted_text = []
            for header, content in context_data.items():
                formatted_text.extend([
                    header,
                    "",  # blank line after header
                    content,
                    "",   # blank line after content
                ])
            
            # Join with newlines but don't strip
            return "\n".join(formatted_text)
        except FileNotFoundError:
            logging.warning(f"Context file {CONTEXT_FILE} not found")
            return ""
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in {CONTEXT_FILE}")
            return ""
        except Exception as e:
            logging.error(f"Error reading context file: {str(e)}")
            return ""

# Main block remains at module level
if __name__ == "__main__":
    # Set up logging
    log_path = Path(BASE_PATH) / 'logs'
    log_path.mkdir(exist_ok=True)
    LOG_FILENAME = log_path / f"agent_conversation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_UTC.log"
    
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            return f"[{datetime.utcnow().strftime('%d/%m/%Y UTC %H:%M:%S')}] - {record.levelname} - {record.getMessage()}"

    handler = logging.FileHandler(LOG_FILENAME)
    handler.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    
    try:
        agent = Agent()
        agent.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"Fatal error occurred. Check logs at: {LOG_FILENAME}")
        sys.exit(1)