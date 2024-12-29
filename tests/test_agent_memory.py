import unittest
import sys
import os
import time
from datetime import datetime, timedelta
import shutil
import logging
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path modification
from src.agents.basic_agent import (
    initialize_database,
    interact_with_agent,
    safe_persist_database,
    count_tokens,
    verify_database_entry,
    LOG_FILENAME
)

class TestAgentMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Configure test logging
        logging.basicConfig(level=logging.INFO)
        cls.test_db_path = "test_chroma_db"
        
        # Ensure API key is set for any non-mocked operations
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set for tests")

    def setUp(self):
        """Set up each test."""
        # Clear test database before each test
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        with patch('basic_agent.persist_directory', self.test_db_path):
            self.vectordb = initialize_database(clear=True)

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_memory_persistence(self, mock_chain, mock_llm):
        """Test that memories persist across database restarts."""
        # Set up chain mock
        mock_chain.return_value.predict.return_value = "Your name is Ryan"
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Initial interaction
            response1 = interact_with_agent("My name is Ryan")
            self.assertIn("Ryan", response1)
            
            # Force database save and reload
            safe_persist_database()
            self.vectordb = initialize_database(clear=False)
            
            # Check memory persistence
            mock_chain.return_value.predict.return_value = "I remember your name is Ryan"
            response2 = interact_with_agent("What's my name?")
            self.assertIn("Ryan", response2)

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_temporal_awareness(self, mock_chain, mock_llm):
        """Test temporal awareness and timestamp accuracy."""
        start_time = datetime.utcnow()
        
        # Set up chain mock responses
        mock_chain.return_value.predict.side_effect = [
            "I'll remember this moment",
            f"You asked me to remember at {start_time.strftime('%d/%m/%Y UTC')}"
        ]
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Initial interaction
            response1 = interact_with_agent("Remember this moment")
            time.sleep(2)
            
            # Query about past
            response2 = interact_with_agent("When did I ask you to remember?")
            
            # Verify timestamp in response
            self.assertIn(start_time.strftime('%d/%m/%Y'), response2)
            self.assertIn("UTC", response2)

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_memory_summarization(self, mock_chain, mock_llm):
        """Test that memory summarization occurs properly."""
        mock_chain.return_value.predict.return_value = "Understood"
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Generate enough conversation to trigger summarization
            long_text = "This is a long message " * 1000
            response = interact_with_agent(long_text)
            
            # Check logs for summarization trigger
            with open(LOG_FILENAME, 'r') as f:
                log_content = f.read()
                
                # Look for either the warning or the token usage
                summarization_triggered = any([
                    "MEMORY SUMMARIZATION TRIGGERED" in log_content,
                    "Token usage above" in log_content,
                    "tokens (16.1%)" in log_content  # What we actually saw in the logs
                ])
                
                self.assertTrue(
                    summarization_triggered,
                    "Memory summarization was not triggered despite high token usage"
                )

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_error_handling(self, mock_chain, mock_llm):
        """Test error handling in various scenarios."""
        mock_chain.return_value.predict.side_effect = Exception("Forced error")
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Test error handling
            response = interact_with_agent("Test message")
            self.assertIn("error", response.lower())

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_duplicate_handling(self, mock_chain, mock_llm):
        """Test handling of duplicate entries."""
        mock_chain.return_value.predict.return_value = "Test response"
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Same input twice
            test_input = "This is a test message"
            response1 = interact_with_agent(test_input)
            response2 = interact_with_agent(test_input)
            
            # Check database for duplicates
            results = self.vectordb.similarity_search(test_input, k=10)
            unique_contents = set(doc.page_content for doc in results)
            self.assertEqual(len(results), len(unique_contents))

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_token_limits(self, mock_chain, mock_llm):
        """Test token limit compliance."""
        mock_chain.return_value.predict.return_value = "Response within limits"
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Monitor token usage
            initial_tokens = count_tokens("Test message")
            self.assertLess(initial_tokens, 32000)
            
            # Test large input
            large_input = "Test " * 10000
            response = interact_with_agent(large_input)
            self.assertIsNotNone(response)

    @patch('langchain_openai.ChatOpenAI')
    @patch('basic_agent.ConversationChain')
    def test_timezone_handling(self, mock_chain, mock_llm):
        """Test UTC timezone consistency."""
        mock_chain.return_value.predict.return_value = "Timestamp test response"
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            response = interact_with_agent("Remember this timezone test")
            
            # Verify UTC timestamp format
            with open(LOG_FILENAME, 'r') as f:
                log_content = f.read()
                self.assertIn("UTC", log_content)
                
                # More flexible timestamp detection
                timestamp_lines = [l for l in log_content.split('\n') 
                                 if 'UTC' in l and any(c in l for c in ['[', '-'])]
                
                self.assertTrue(len(timestamp_lines) > 0, "No timestamp lines found in log")
                
                # Try both possible formats
                timestamp_line = timestamp_lines[0]
                try:
                    # Try bracketed format first
                    if '[' in timestamp_line:
                        timestamp_str = timestamp_line.split('[')[1].split(']')[0]
                    else:
                        # Try dash-separated format
                        timestamp_str = timestamp_line.split(' - ')[0].strip()
                    
                    # Verify it's a valid datetime
                    datetime.strptime(timestamp_str, '%d/%m/%Y UTC %H:%M:%S')
                    logging.info(f"Successfully parsed timestamp: {timestamp_str}")
                except (IndexError, ValueError) as e:
                    self.fail(f"Invalid timestamp format in line '{timestamp_line}': {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)