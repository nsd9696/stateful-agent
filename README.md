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
- 📚 Research lab collections management
- 📖 Google Scholar paper tracking and crawling
- 🔬 Paper recommendation based on research interests
- 📝 Contextual paper summarization with related research

## New Features: Paper Recommendation and Summary

The agent now supports robust academic paper management and recommendation features:

### Research Lab Management

- Create lab collections with persistent information (name, institution, leader, members, etc.)
- Add lab members with their Google Scholar profiles
- Track papers published by lab members

### Paper Collection and Recommendation

- Automatically crawl Google Scholar pages of lab members to collect their arXiv papers
- Check for new papers by lab members during conversations
- Store PDF documents in the data directory with proper organization
- Recommend relevant papers from arXiv based on the lab's research interests and time period
- Save recommended papers and their embeddings to prevent duplication

### Paper Summarization

- Generate comprehensive paper summaries for specific lab member papers
- Utilize complete paper content for more thorough and accurate summaries of target papers
- Extract semantic sections (introduction, conclusion) from LaTeX source files of related papers when available
- Include contextual information from related papers in the lab collection
- Draw insights from both lab papers and recommended papers
- Provide academic-style summaries with key findings, methodologies, and relationships to existing research

## Project Structure

```
stateful-agent/
├── stateful_agent/           # Main package directory
│   ├── tools/               # Tool implementations
│   │   ├── sqlite.py        # Entity database operations
│   │   ├── chromadb.py      # Vector database operations
│   │   └── paper_crawler.py # Paper collection and recommendation tools
│   ├── agent.py             # Core agent implementation
│   ├── paper_recommendation_agent.py # Specialized paper recommendation agent
│   ├── data/                # Data storage directory
│   │   ├── <lab_name>/      # Lab-specific paper PDFs
│   │   └── recommendation/  # Recommended paper PDFs
│   ├── .env                 # Environment configuration
│   └── .secrets.toml        # Secret configuration (not tracked)
└── frontend/                # Frontend implementation
```

## Prerequisites

- Python 3.11 or higher
- OpenAI API key (gpt-4o model for paper summarization and embeddings)
- Internet connection for accessing Google Scholar and arXiv
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
mkdir -p data/recommendation
```

2. Run the agent:

```bash
uv run python paper_recommendation_agent.py
```

3. Example interactions:

```
# Create a new research lab
> Create a lab called vision_research_lab at University of California, Berkeley, with leader Jitendra Malik

# Add members with their Google Scholar profiles
> Add member Haozhi Qi with scholar URL https://scholar.google.com/citations?user=iyVHKkcAAAAJ&hl=en to vision_research_lab

# Add research areas for the lab
> Add computer vision, machine learning and robotics as research areas for vision_research_lab

# Add website and description for the lab
> Add https://people.eecs.berkeley.edu/~malik/ as the website for vision_research_lab, and add description for the lab: Vision Intelligence

# Crawl Google Scholar for papers by lab members
> Collect papers from vision_research_lab members

# Stay updated with the lab
> Check new papers for vision_research_lab

# Get paper recommendations
> Recommend 5 papers from the last 30 days related to vision_research_lab

# Generate a paper summary
> Summarize the latest paper by Haozhi Qi from vision_research_lab

> Summarize the latest paper from vision_research_lab

# Delete the lab
> Delete the lab vision_research_lab
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
- [arXiv](https://arxiv.org/) for access to research papers
- [Zotero-arXiv-Daily](https://github.com/TideDra/zotero-arxiv-daily) for inspiration on paper recommendation