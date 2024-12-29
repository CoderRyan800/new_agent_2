from datetime import datetime
import json
import os
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

class MemoryUtils:
    _embeddings = None
    
    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            cls._embeddings = OpenAIEmbeddings()
        return cls._embeddings

    @staticmethod
    def safe_calculator(expression):
        """Calculator utility function"""
        try:
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error calculating: {str(e)}"

    @staticmethod
    def edit_context(command, context_file):
        """Edit context file utility function"""
        try:
            header, content = command.split('|', 1)
            header = header.strip().lower()
            
            if header not in ['persona', 'human', 'context_notes']:
                return "Error: Header must be one of: persona, human, context_notes"
                
            try:
                with open(context_file, 'r') as f:
                    context_data = json.loads(f.read())
            except FileNotFoundError:
                context_data = {}
            except json.JSONDecodeError:
                return "Error: Context file contains invalid JSON"
                
            for key in ['persona', 'human', 'context_notes']:
                if key not in context_data:
                    context_data[key] = ""
                    
            context_data[header] = content
                
            with open(context_file, 'w') as f:
                json.dump(context_data, f, indent=4)
                
            return f"Success: Updated {header}"
            
        except ValueError:
            return "Error: Invalid format. Use 'header|content'"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def initialize_manual_memory(db_path):
        """Initialize manual memory database"""
        try:
            if os.path.exists(db_path):
                embeddings = OpenAIEmbeddings()
                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                count = vectordb._collection.count()
                logging.info(f"Manual Memory: Checked existing system. Found {count} entries")
                return f"Manual memory system already exists with {count} entries"
            else:
                embeddings = OpenAIEmbeddings()
                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                logging.info("Manual Memory: Initialized new system")
                return "Manual memory system initialized successfully"
        except Exception as e:
            logging.error(f"Manual Memory Error: Failed to initialize - {str(e)}")
            return f"Error initializing manual memory: {str(e)}"

    @staticmethod
    def write_to_manual_memory(entry, db_path):
        """Write to manual memory database"""
        try:
            title, content = entry.split('|', 1)
            title = title.strip()
            content = content.strip()
            
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            timestamp = datetime.utcnow()
            
            vectordb.add_texts(
                texts=[content],
                metadatas=[{
                    "title": title,
                    "timestamp": timestamp.isoformat(),
                    "timestamp_readable": timestamp.strftime('%d/%m/%Y UTC %H:%M:%S')
                }]
            )
            
            logging.info(f"Manual Memory: Stored new memory - Title: {title}")
            logging.info(f"Manual Memory: Content: {content}")
            return f"Successfully stored memory: {title}"
        except ValueError:
            logging.error("Manual Memory Error: Invalid format for memory entry")
            return "Error: Entry must be in format 'title|content'"
        except Exception as e:
            logging.error(f"Manual Memory Error: Failed to store memory - {str(e)}")
            return f"Error storing memory: {str(e)}"

    @staticmethod
    def search_manual_memory(query, db_path):
        """Search manual memory database"""
        try:
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            if vectordb._collection.count() == 0:
                logging.info("Manual Memory: Search attempted on empty database")
                return "Manual memory is empty"
                
            logging.info(f"Manual Memory: Searching for: {query}")
            results = vectordb.similarity_search(query, k=3)
            
            if not results:
                logging.info("Manual Memory: No results found for search")
                return "No relevant memories found"
                
            formatted_results = []
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                title = metadata.get('title', 'Untitled')
                timestamp = metadata.get('timestamp_readable', 'No timestamp')
                formatted_results.append(f"{i}. {title} [{timestamp}]:\n{doc.page_content}\n")
                
            result_text = "\n".join(formatted_results)
            logging.info(f"Manual Memory: Search results:\n{result_text}")
            return result_text
            
        except Exception as e:
            logging.error(f"Manual Memory Error: Search failed - {str(e)}")
            return f"Error searching memories: {str(e)}"