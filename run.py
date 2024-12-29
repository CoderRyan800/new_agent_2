#!/usr/bin/env python3
"""
Entry point for new_agent_2
"""
import os
import sys
from pathlib import Path
import logging

# Disable pdb
import pdb
pdb.set_trace = lambda: None

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Setup logging directory
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

from src.agents.basic_agent import Agent

def main():
    """Main entry point for the agent."""
    try:
        print("\nInitializing agent...")
        agent = Agent()
        print("Agent initialized successfully. Starting conversation...\n")
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        logging.exception("Fatal error occurred")
        sys.exit(1)

if __name__ == "__main__":
    main() 