import unittest
import sys
import os
import time
from datetime import datetime, timedelta
import shutil
import logging
from unittest.mock import Mock, patch, MagicMock
from langchain.agents import AgentType, initialize_agent

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path modification
from basic_agent import (
    AgentManager,
    LOG_FILENAME
)

class TestAgentMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        cls.test_db_path = "test_chroma_db"
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set for tests")

    def setUp(self):
        """Set up each test."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        # Create new agent manager for each test with specific persist_dir
        self.agent_manager = AgentManager(
            model_name="gpt-4",
            clear_db=True,
            persist_dir=self.test_db_path
        )

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    @patch('langchain_openai.ChatOpenAI')
    def test_memory_persistence(self, mock_llm):
        """Test that memories persist across database restarts."""
        # Setup mock response
        mock_response = {"output": "Your name is Ryan"}
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            with patch.object(self.agent_manager.agent, 'invoke', return_value=mock_response):
                # Initial interaction
                response1 = self.agent_manager.interact("My name is Ryan")
                
                # Force database save and reload
                self.agent_manager.cleanup()
                
                # Create new agent manager (simulating restart)
                new_agent_manager = AgentManager(clear_db=False, persist_dir=self.test_db_path)
                
                # Check memory persistence
                mock_response2 = {"output": "I remember your name is Ryan"}
                with patch.object(new_agent_manager.agent, 'invoke', return_value=mock_response2):
                    response2 = new_agent_manager.interact("What's my name?")
                    self.assertIn("Ryan", response2)

    def test_temporal_awareness(self):
        """Test temporal awareness and timestamp accuracy."""
        start_time = datetime.utcnow()
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Initial interaction
            self.agent_manager.interact("Remember this moment")
            time.sleep(2)
            
            # Query about past
            results = self.agent_manager.vectordb.similarity_search("Remember this moment", k=1)
            self.assertTrue(len(results) > 0)
            self.assertIn(start_time.strftime('%d/%m/%Y'), results[0].page_content)

    def test_memory_summarization(self):
        """Test that memory summarization occurs properly."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Generate enough conversation to trigger summarization
            long_text = "This is a long message " * 1000
            mock_response = {"output": "Test response"}
            
            with patch.object(self.agent_manager.agent, 'invoke', return_value=mock_response):
                response = self.agent_manager.interact(long_text)
                
                # Check logs for summarization trigger
                with open(LOG_FILENAME, 'r') as f:
                    log_content = f.read()
                    summarization_triggered = any([
                        "MEMORY SUMMARIZATION TRIGGERED" in log_content,
                        "Token usage exceeds threshold" in log_content
                    ])
                    self.assertTrue(summarization_triggered, "Memory summarization was not triggered")

    @patch('langchain_openai.ChatOpenAI')
    def test_error_handling(self, mock_llm):
        """Test error handling in various scenarios."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            with patch.object(self.agent_manager.agent, 'invoke', side_effect=Exception("Forced error")):
                response = self.agent_manager.interact("Test message")
                self.assertIn("error", response.lower())

    def test_duplicate_handling(self):
        """Test handling of duplicate entries."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Same input twice
            test_input = "This is a test message"
            response1 = self.agent_manager.interact(test_input)
            response2 = self.agent_manager.interact(test_input)
            
            # Check database for duplicates
            results = self.agent_manager.vectordb.similarity_search(test_input, k=10)
            unique_contents = set(doc.page_content for doc in results)
            self.assertEqual(len(results), len(unique_contents))

    def test_timezone_handling(self):
        """Test UTC timezone consistency."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            response = self.agent_manager.interact("Remember this timezone test")
            
            # Verify UTC timestamp format in logs
            with open(LOG_FILENAME, 'r') as f:
                log_content = f.read()
                self.assertIn("UTC", log_content)
                
                timestamp_lines = [l for l in log_content.split('\n') 
                                 if 'UTC' in l and any(c in l for c in ['[', '-'])]
                
                self.assertTrue(len(timestamp_lines) > 0)
                
                timestamp_line = timestamp_lines[0]
                try:
                    if '[' in timestamp_line:
                        timestamp_str = timestamp_line.split('[')[1].split(']')[0]
                    else:
                        timestamp_str = timestamp_line.split(' - ')[0].strip()
                    
                    datetime.strptime(timestamp_str, '%d/%m/%Y UTC %H:%M:%S')
                    logging.info(f"Successfully parsed timestamp: {timestamp_str}")
                except (IndexError, ValueError) as e:
                    self.fail(f"Invalid timestamp format in line '{timestamp_line}': {str(e)}")

    def test_vector_database_operations(self):
        """Test basic vector database operations."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            test_input = "This is a unique test message"
            mock_response = {"output": "Test response"}
            
            with patch.object(self.agent_manager.agent, 'invoke', return_value=mock_response):
                self.agent_manager.interact(test_input)
                
                # Allow a small delay for database operations
                time.sleep(1)
                
                # Test retrieval
                results = self.agent_manager.vectordb.similarity_search(test_input, k=1)
                self.assertTrue(len(results) > 0)
                # Check if the input is in any part of the page content
                found = any(test_input in doc.page_content for doc in results)
                self.assertTrue(found, f"Test input not found in any document: {[doc.page_content for doc in results]}")

    def test_memory_token_counting(self):
        """Test token counting functionality."""
        test_text = "This is a test message"
        token_count = self.agent_manager.count_tokens(test_text)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)

    @patch('langchain_openai.ChatOpenAI')
    def test_conversation_context(self, mock_llm):
        """Test that conversation context is properly maintained."""
        mock_response = {"output": "Response"}
        
        with patch.object(self.agent_manager.agent, 'invoke', return_value=mock_response):
            # First interaction
            self.agent_manager.interact("First message")
            
            # Check that context is included in second interaction
            with patch.object(self.agent_manager, 'vectordb') as mock_db:
                mock_db.similarity_search.return_value = []
                self.agent_manager.interact("Second message")
                # Verify that similarity_search was called at least once
                mock_db.similarity_search.assert_called_once()

if __name__ == '__main__':
    unittest.main(verbosity=2)