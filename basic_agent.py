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

# Get the OpenAI API key from the environment
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

os.environ["OPENAI_API_KEY"] = api_key

# Set the max token limit
MAX_TOKENS = 4096

# Define the path for the Chroma database
persist_directory = os.path.join(os.getcwd(), "chroma_db")

# Initialize the Chroma vector store
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

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
memory = ConversationBufferMemory()

# After initializing vectordb
print(f"Chroma database initialized. Number of elements: {vectordb._collection.count()}")

def interact_with_agent(user_input):
    """Interacts with the agent and manages conversation history."""
    try:
        # Get the current conversation from the buffer
        conversation = memory.buffer

        # Summarize and manage the conversation
        updated_conversation = summarize_conversation(conversation, user_input)

        # Add only the new information to the vector database
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_text = updated_conversation.split("\n")[-1]  # Get only the latest addition
        texts = text_splitter.split_text(new_text)
        vectordb.add_texts(texts)

        # Print the number of elements in the database after adding new texts
        print(f"Added new texts. Number of elements: {vectordb._collection.count()}")

        # Retrieve relevant past conversations
        db_size = len(vectordb.get())  # Get the current size of the database
        k = min(2, db_size)  # Use 2 or the database size, whichever is smaller
        relevant_history = vectordb.similarity_search(user_input, k=k)
        
        # Combine current conversation with relevant history
        full_context = (
            "Relevant past information:\n" +
            "\n".join([doc.page_content for doc in relevant_history]) +
            "\n\nCurrent conversation:\n" + updated_conversation
        )

        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant named John. The user's name is Ryan."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Create the conversation chain
        conversation = ConversationChain(
            llm=ChatOpenAI(temperature=0),
            prompt=prompt,
            memory=ConversationBufferMemory(return_messages=True)
        )

        # Run the conversation with the full context
        response = conversation.predict(input=full_context)

        # Update the conversation buffer with the response
        memory.save_context({"input": user_input}, {"output": response})

        # After adding new texts or performing a search, persist the database
        vectordb.persist()

        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request."

# Example interaction loop
while True:
    user_input = input("You: ")
    response = interact_with_agent(user_input)
    print(f"Agent: {response}")
