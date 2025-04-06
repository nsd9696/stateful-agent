# Stateful Agent

A research paper management and LinkedIn publishing agent.

## Environment Variables Setup

To use this agent, you need to set up the following environment variables in a `.env` file in the `stateful_agent` directory:

```
# OpenAI Configuration
OPENAI_API_KEY_AGENT=your_agent_api_key
OPENAI_API_KEY_SUMMARIZER=your_summarizer_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Database Paths
CHROMA_PERSIST_DIRECTORY=./chroma_langchain_db
SQLITE_DB_PATH=./sqlite_langchain_db.db
DEFAULT_DATA_DIR=./data

# LinkedIn Configuration
LINKEDIN_USER_ID=your_linkedin_user_id
LINKEDIN_ACCESS_TOKEN=your_linkedin_access_token
```

### API Key Usage

The codebase uses the following API keys:

1. `OPENAI_API_KEY_AGENT`: Used by the main agent for conversation, reasoning, and embeddings
2. `OPENAI_API_KEY_SUMMARIZER`: Used only by the paper_scraper.py for paper summarization and LinkedIn post generation

## Usage

Run the agent with:

```bash
python stateful_agent/agent.py
```

## Features

- Create and manage research lab collections
- Track papers from lab members
- Recommend relevant papers
- Generate comprehensive paper summaries
- Publish paper summaries to LinkedIn with PDF attachments
