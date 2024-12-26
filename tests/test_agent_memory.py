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
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY must be set for tests")
        cls.test_db_path = os.path.join(os.getcwd(), "test_chroma_db")

    def setUp(self):
        """Set up test fixtures."""
        # Clear the test database before each test
        if os.path.exists(self.test_db_path):
            # First remove read-only flag if it exists
            if os.path.exists(os.path.join(self.test_db_path, "chroma.sqlite3")):
                os.chmod(os.path.join(self.test_db_path, "chroma.sqlite3"), 0o666)
            shutil.rmtree(self.test_db_path)
        
        # Create a new agent manager with mocked components
        with patch('langchain_openai.ChatOpenAI') as mock_llm:
            self.agent_manager = AgentManager(
                clear_db=True,
                persist_dir=self.test_db_path
            )
            # Replace the agent with a mock
            self.agent_manager.agent = MagicMock()
            self.agent_manager.agent.run = MagicMock()
            self.agent_manager.agent.invoke = MagicMock()

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    @patch('langchain_openai.ChatOpenAI')
    def test_memory_persistence(self, mock_llm):
        """Test that memories persist across database restarts."""
        # Setup mock response
        mock_response = "Your name is Ryan"
        self.agent_manager.agent.run.return_value = mock_response
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Initial interaction
            response1 = self.agent_manager.interact("My name is Ryan")
            
            # Force database save and reload
            self.agent_manager.cleanup()
            
            # Create new agent manager (simulating restart)
            new_agent_manager = AgentManager(clear_db=False, persist_dir=self.test_db_path)
            new_agent_manager.agent = MagicMock()
            new_agent_manager.agent.run.return_value = "I remember your name is Ryan"
            
            # Check memory persistence
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
            long_text = "This is a long message. " * 1000
            self.agent_manager.agent.run.return_value = "Test response"
            
            # Capture log output directly
            with self.assertLogs(level='INFO') as log_context:
                self.agent_manager.interact(long_text)
                
                # Check logs for token count
                log_output = '\n'.join(log_context.output)
                self.assertIn("Memory Allocation", log_output)
                self.assertIn("Total usage", log_output)
                
                # Verify that a large amount of tokens were processed
                token_count = self.agent_manager.count_tokens(long_text)
                self.assertGreater(token_count, 1000, 
                                 f"Token count {token_count} is too low for summarization test")

    @patch('langchain_openai.ChatOpenAI')
    def test_error_handling(self, mock_llm):
        """Test error handling in various scenarios."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Force an error by making run raise an exception
            self.agent_manager.agent.run.side_effect = Exception("Test error")
            response = self.agent_manager.interact("Test message")
            self.assertIn("error", response.lower())

    def test_duplicate_handling(self):
        """Test handling of duplicate entries."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Clear any existing interactions
            if os.path.exists(self.test_db_path):
                shutil.rmtree(self.test_db_path)
            
            # Same input twice
            test_input = "This is a test message for duplicate handling"
            self.agent_manager.agent.run.return_value = "Test response"
            
            # First interaction
            self.agent_manager.interact(test_input)
            time.sleep(1)  # Allow for database write
            
            # Second interaction with same input
            self.agent_manager.interact(test_input)
            time.sleep(1)  # Allow for database write
            
            # Check database for duplicates
            results = self.agent_manager.vectordb.similarity_search(test_input, k=10)
            
            # Print debug information
            print("\nDatabase contents for duplicate test:")
            for doc in results:
                print(f"Document: {doc.page_content}")
            
            # Extract timestamps and messages to verify uniqueness
            interactions = []
            for doc in results:
                for line in doc.page_content.split('\n'):
                    if 'User: ' in line:
                        timestamp = doc.page_content.split('\n')[0]  # Get timestamp from first line
                        message = line.split('User: ')[1]
                        interactions.append((timestamp, message))
            
            # Verify that we don't have the same message at the same timestamp
            unique_interactions = set(interactions)
            self.assertEqual(
                len(interactions), 
                len(unique_interactions),
                f"Duplicates found in interactions: {interactions}"
            )

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
            self.agent_manager.agent.run.return_value = "Test response"
            
            # Clear any existing interactions
            if os.path.exists(self.test_db_path):
                shutil.rmtree(self.test_db_path)
            
            # Perform interaction
            response = self.agent_manager.interact(test_input)
            time.sleep(1)  # Allow for database write
            
            # Test retrieval
            results = self.agent_manager.vectordb.similarity_search(test_input, k=1)
            self.assertTrue(len(results) > 0, "No results found in vector database")
            
            # Print debug information
            print("\nExpected input:", test_input)
            print("Response:", response)
            print("\nDatabase contents:")
            for doc in results:
                print(f"Document: {doc.page_content}")
            
            # Check if the test input appears in any document
            found = any(test_input in doc.page_content for doc in results)
            self.assertTrue(found, 
                           f"Test input '{test_input}' not found in documents")

    def test_memory_token_counting(self):
        """Test token counting functionality."""
        test_text = "This is a test message"
        token_count = self.agent_manager.count_tokens(test_text)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)

    def test_conversation_context(self):
        """Test that conversation context is properly maintained."""
        # Setup mock response
        self.agent_manager.agent.run.return_value = "Test response"
        
        # First interaction
        first_message = "First message"
        self.agent_manager.interact(first_message)
        
        # Second interaction should trigger a search
        second_message = "Second message"
        with patch.object(self.agent_manager.vectordb, 'similarity_search') as mock_search:
            mock_search.return_value = []  # Return empty results
            self.agent_manager.interact(second_message)
            # Verify that similarity_search was called at least once
            mock_search.assert_called()
            # Verify the search input contains our message
            call_args = mock_search.call_args[0][0]
            self.assertIn(second_message, call_args)

if __name__ == '__main__':
    unittest.main(verbosity=2)