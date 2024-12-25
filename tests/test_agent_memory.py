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
        mock_llm.return_value.predict.return_value = "Your name is Ryan"
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Initial interaction
            response1 = self.agent_manager.interact("My name is Ryan")
            self.assertIn("Ryan", response1)
            
            # Force database save and reload
            self.agent_manager.cleanup()
            
            # Create new agent manager (simulating restart)
            new_agent_manager = AgentManager(clear_db=False)
            
            # Check memory persistence
            mock_llm.return_value.predict.return_value = "I remember your name is Ryan"
            response2 = new_agent_manager.interact("What's my name?")
            self.assertIn("Ryan", response2)

    @patch('langchain_openai.ChatOpenAI')
    def test_temporal_awareness(self, mock_llm):
        """Test temporal awareness and timestamp accuracy."""
        start_time = datetime.utcnow()
        
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Initial interaction
            response1 = self.agent_manager.interact("Remember this moment")
            time.sleep(2)
            
            # Query about past
            response2 = self.agent_manager.interact("When did I ask you to remember?")
            
            # Verify timestamp in response
            self.assertIn(start_time.strftime('%d/%m/%Y'), response2)
            self.assertIn("UTC", response2)

    def test_memory_summarization(self):
        """Test that memory summarization occurs properly."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            # Generate enough conversation to trigger summarization
            long_text = "This is a long message " * 1000
            response = self.agent_manager.interact(long_text)
            
            # Check logs for summarization trigger
            with open(LOG_FILENAME, 'r') as f:
                log_content = f.read()
                summarization_triggered = any([
                    "MEMORY SUMMARIZATION TRIGGERED" in log_content,
                    "Token usage above" in log_content,
                    "tokens (16.1%)" in log_content
                ])
                self.assertTrue(summarization_triggered)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        with patch('basic_agent.persist_directory', self.test_db_path):
            with patch.object(self.agent_manager.agent, 'run', side_effect=Exception("Forced error")):
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

if __name__ == '__main__':
    unittest.main(verbosity=2)