from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
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
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler, console_handler]
)

# Get the OpenAI API key from the environment
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

os.environ["OPENAI_API_KEY"] = api_key

# Set the max token limit
MAX_TOKENS = 16384  # For GPT-4 32k version

# Define the path for the Chroma database
persist_directory = os.path.join(os.getcwd(), "chroma_db")

def initialize_database(clear=False):
    if clear and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Cleared existing database at {persist_directory}")
    
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Chroma database initialized. Number of elements: {vectordb._collection.count()}")
    return vectordb

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the AI agent with optional database clearing.")
parser.add_argument('--clear-db', action='store_true', help='Clear the existing database before initializing')
args = parser.parse_args()

# Initialize the database (clear only if --clear-db flag is set)
vectordb = initialize_database(clear=args.clear_db)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    """Count the number of tokens in a string."""
    return len(tokenizer.encode(text))

def summarize_conversation(conversation, new_input):
    """Summarizes the conversation and manages token length."""
    current_tokens = count_tokens(conversation)
    summarization_limit = int(0.75 * MAX_TOKENS)

    if current_tokens > summarization_limit:
        llm = OpenAI(temperature=0)
        summary = llm(f"Summarize the following conversation, preserving key points and context: {conversation}")
        new_conversation = summary + "\n\nRecent messages:\n" + conversation[-500:]  # Keep last 500 chars for continuity
        new_conversation += "\n" + new_input
        return new_conversation
    else:
        return conversation + "\n" + new_input

# Initialize the conversation buffer memory
memory = ConversationBufferMemory(return_messages=True)

def get_text_hash(text):
    """Generate a hash for the given text."""
    return hashlib.md5(text.encode()).hexdigest()

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant named John. The user's name is Ryan."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

def interact_with_agent(user_input):
    try:
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

        # Log the full context being sent to the agent
        logging.info(f"Full context being sent to agent:\n{full_context}")

        conversation = ConversationChain(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=MAX_TOKENS),
            prompt=prompt,
            memory=memory
        )

        # Get the response from the agent
        response = conversation.predict(input=full_context)
        
        # Log the response
        logging.info(f"Agent response:\n{response}")

        # Now that we have the response, prepare the new interaction text
        new_interaction = f"User: {user_input}\nAgent: {response}"

        # Generate a hash for the new interaction
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
            vectordb.add_texts(texts, metadatas=metadatas)
            logging.info("=== NEW UNIQUE INTERACTION - ADDING TO DATABASE ===")
            logging.info(f"Number of chunks added: {len(texts)}")
            logging.info("New interaction content:")
            logging.info(f"{new_interaction}")
            logging.info("============================================")
        
        # Update the conversation buffer with the response
        memory.save_context({"input": user_input}, {"output": response})
        
        # Log the current state of the conversation memory
        memory_vars = memory.load_memory_variables({})
        logging.info("Current conversation memory buffer contents:")
        logging.info(f"Memory variables:\n{memory_vars}")

        return response
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return "I'm sorry, but I encountered an error while processing your request."
    finally:
        # Always persist the database, even if an error occurred
        vectordb.persist()
        logging.info(f"Database persisted. Current number of elements: {vectordb._collection.count()}")

# Example interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Gracefully shutting down...")
        vectordb.persist()
        print(f"Final database state. Number of elements: {vectordb._collection.count()}")
        break
    response = interact_with_agent(user_input)
    print(f"Agent: {response}")

# Final persistence after the loop
vectordb.persist()
print(f"Final database state. Number of elements: {vectordb._collection.count()}")

if __name__ == "__main__":
    # Your main loop or function calls here
    pass

def signal_handler(sig, frame):
    print("\nGracefully shutting down...")
    vectordb.persist()
    print(f"Final database state. Number of elements: {vectordb._collection.count()}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def cleanup():
    print("Cleaning up...")
    vectordb.persist()
    # If Chroma has a close method, call it here
    # vectordb._client.close()

atexit.register(cleanup)
