# Stateful-Agent

A sophisticated agent system that maintains state and context through the combination of persistent entity storage (SQLite) and vector-based memory (ChromaDB).

## Overview

Stateful-Agent is designed to provide a robust solution for maintaining conversational context and user information across interactions. It leverages two different types of databases:

- **SQLite**: For structured data storage and retrieval of user information and persistent state
- **ChromaDB**: For vector-based storage enabling semantic search and contextual memory

## Features

- 🔄 Persistent user data storage with SQLite
- 🧠 Semantic memory capabilities using ChromaDB vector database
- 🔍 Case-insensitive user handling
- 🚫 Duplicate prevention for user entries
- ✅ Structured data validation and management
- 📄 PDF document processing and analysis
- 🔗 Integration with various external tools (GitHub, Slack, Google)

## Project Structure

```
stateful-agent/
├── stateful_agent/           # Main package directory
│   ├── tools/               # Tool implementations
│   │   ├── sqlite.py        # Entity database operations
│   │   └── chromadb.py      # Vector database operations
│   ├── agent.py             # Core agent implementation
│   ├── data/                # Data storage directory
│   ├── .env                 # Environment configuration
│   └── .secrets.toml        # Secret configuration (not tracked)
└── frontend/                # Frontend implementation
```

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- (Optional) GitHub, Slack, or Google credentials for additional features

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nsd9696/stateful-agent.git
cd stateful-agent
```

2. Create and activate a virtual environment:

```bash
cd stateful_agent
```

- .env

```bash
pip install uv
uv pip install -e ".[dev]"
```

## Configuration

1. Create necessary configuration files in the `stateful_agent` directory:

### .env

```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=YOUR_OPENAI_KEY

CHROMA_PERSIST_DIRECTORY=./chroma_langchain_db
SQLITE_DB_PATH=./sqlite_langchain_db.db
DEFAULT_DATA_DIR=./data
```

### .secrets.toml (Optional)

```toml
[git.github]
github_token = "YOUR_GITHUB_TOKEN"

[auth.slack]
client_id = "SLACK_APP_CLIENT_ID"
client_secret = "SLACK_APP_CLIENT_SECRET"

[auth.google]
client_id = "GOOGLE_CLIENT_ID"
client_secret = "GOOGLE_CLIENT_SECRET"
```

## Usage

1. Prepare the environment:

```bash
cd stateful_agent
mkdir -p data
```

2. Run the agent in terminal mode:

```bash
uv run stateful-agent deploy-agent --file agent.py --mode terminal
```

3. Run the agent in web mode:

```bash
uv run stateful-agent deploy-agent --file agent.py --mode web
```

3. Example interactions:

```
# Create a new user
> Make john's user data

# Create a collection for the user
> Make john's collection

# Add a document to the user's collection
> Add /path/to/document.pdf to john's collection

# Query the agent
> What are the main topics in john's documents?
```

## Development

- Run tests: `pytest`
- Format code: `black . && isort .`
- Type checking: `mypy .`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the core agent capabilities
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [OpenAI](https://openai.com/) for embedding and completion APIs
