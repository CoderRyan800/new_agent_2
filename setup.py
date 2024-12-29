from setuptools import setup, find_packages

setup(
    name="new_agent_2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'langchain==0.3.4',
        'langchain-openai==0.2.3',
        'langchain-community==0.3.3',
        'langchain-core==0.3.12',
        'langchain-chroma==0.1.4',
        'langchain-text-splitters==0.3.0',
        'chromadb>=0.4.22',
        'tiktoken>=0.5.1',
        'python-dotenv>=0.19.0',
        'openai>=1.1.0'
    ],
    python_requires='>=3.8',
) 