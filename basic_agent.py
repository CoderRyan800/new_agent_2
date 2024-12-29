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
import json

# Constants
MAX_CONVERSATION_TOKENS = 8000
MODEL_SELECTION = "gpt-4"
PERSIST_DIRECTORY = "chroma_db"
CONTEXT_FILE = "context_memory.json"

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

def safe_calculator(expression):
    try:
        # Only allow basic math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

def edit_context(command):
    """
    Edit the context file that defines your persona, your knowledge about the human, and your working notes. Command should be in format: 'header|content'
    Only handles three standard headers: persona, human, and context_notes
    """
    try:
        header, content = command.split('|', 1)
        header = header.strip().lower()
        
        # Validate header
        if header not in ['persona', 'human', 'context_notes']:
            return "Error: Header must be one of: persona, human, context_notes"
            
        # Read current context
        try:
            with open(CONTEXT_FILE, 'r') as f:
                context_data = json.loads(f.read())
        except FileNotFoundError:
            context_data = {}
        except json.JSONDecodeError:
            return "Error: Context file contains invalid JSON"
            
        # Ensure all standard keys exist
        for key in ['persona', 'human', 'context_notes']:
            if key not in context_data:
                context_data[key] = ""
                
        # Update the specified section
        context_data[header] = content
            
        # Write back to file
        with open(CONTEXT_FILE, 'w') as f:
            json.dump(context_data, f, indent=4)
            
        return f"Success: Updated {header}"
        
    except ValueError:
        return "Error: Invalid format. Use 'header|content'"
    except Exception as e:
        return f"Error: {str(e)}"

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

        # Add tools setup before conversation chain initialization
        self.tools = [
            Tool(
                name="Calculator",
                func=safe_calculator,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression using numbers and basic operators (+, -, *, /)."
            ),
            Tool(
                name="EditContext",
                func=edit_context,
                description="Edit the context file. Use format 'header|content' where header must be one of: persona, human, context_notes. Example: 'persona|I am a helpful AI assistant'. Use this to update your persona, your knowledge about the human, or your working notes."
            )
        ]

        # Create the prompt with required variables for ReAct
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with perfect memory recall and access to tools. At the start of each interaction, you will see three important context sections:

1. "persona": This defines who you are and how you should behave. You should always act according to this persona. You can edit this using the EditContext tool if you need to update your personality or role.

2. "human": This describes the human you're working with. Use this to better understand and assist your user. You can update this section using EditContext as you learn more about them.

3. "context_notes": This is your notepad for important facts or observations. You can edit these notes using the EditContext tool, but keep them focused and brief. Use this as your working memory for important information.

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

        self.agent = create_react_agent(
            llm=ChatOpenAI(model_name=MODEL_SELECTION, temperature=0),
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
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
            
            context_file_content = self.read_context_file()
            relevant_history = self.vectordb.similarity_search(user_input, k=3) if self.vectordb._collection.count() > 0 else []
            
            context = context_file_content
            if relevant_history:
                context += "\n\nPast context:\n" + "\n".join(doc.page_content for doc in relevant_history)
            context += f"\n\n[{timestamp_str}]\nUser: {user_input}"

            logging.info(f"Context: {context}")
            
            # Use agent_executor
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

# Update the main block
if __name__ == "__main__":
    agent = Agent()
    agent.run()
