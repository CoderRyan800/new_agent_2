#!/usr/bin/env python3
"""
Entry point for new_agent_2
"""
import os
import sys
from pathlib import Path

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
        agent = Agent()
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 