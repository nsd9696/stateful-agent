import os

from hyperdock_fileio import initialize_dock as fileio_dock
from hyperpocket.tool import from_dock
from hyperpocket_langchain import PocketLangchain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import (ConversationBufferMemory,
                              ConversationSummaryMemory)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools.chromadb import (add_pdf_documents, create_collection,
                            query_collection)
from tools.paper_crawler import (check_new_papers, crawl_scholar_papers,
                                 generate_paper_summary, recommend_papers,
                                 crawl_semantic_scholar, check_new_papers_alt,
                                 summarize_latest_author_paper)
from tools.sqlite import (add_lab_member, create_lab, get_all_labs,
                          get_lab_info, get_user_data, insert_user_data,
                          update_lab_website, update_lab_description, add_research_area)
from tools.linkedin_publisher import publish_linkedin_post, publish_paper_to_linkedin, search_and_publish_paper
from tools.paper_scraper import scrape_papers, save_papers_to_db, create_linkedin_post_from_paper
from model import StatefulAgentExecutor

class LinkedInAgent(StatefulAgentExecutor):
    """A specialized agent for LinkedIn publishing operations.
    
    This agent extends the StatefulAgentExecutor to provide LinkedIn-specific functionality
    for publishing academic papers and managing research content. It includes tools for:
    - Paper management and summarization
    - Research lab management
    - LinkedIn post creation and publishing
    - PDF document handling
    - Database operations
    
    The agent uses LangChain for conversation management and OpenAI's GPT-4 for
    natural language processing and content generation.
    """
    executor: AgentExecutor

    @classmethod
    def create_agent(cls, pocket: PocketLangchain = None) -> AgentExecutor:
        """Create a new LinkedIn agent instance with all necessary tools.
        
        Args:
            pocket (PocketLangchain, optional): A pre-configured pocket of tools. If None,
                a new pocket will be created with all required tools.
                
        Returns:
            AgentExecutor: A configured agent executor ready for use.
            
        Raises:
            ValueError: If the OPENAI_API_KEY_AGENT environment variable is not set.
        """
        if pocket is None:
            pocket = PocketLangchain(
                tools=[
                    create_collection,
                    insert_user_data,
                    get_user_data,
                    create_lab,
                    get_lab_info,
                    add_lab_member,
                    get_all_labs,
                    update_lab_website,
                    update_lab_description,
                    add_research_area,
                    *from_dock(fileio_dock()),
                    add_pdf_documents,
                    query_collection,
                    crawl_scholar_papers,
                    check_new_papers,
                    crawl_semantic_scholar,
                    check_new_papers_alt,
                    recommend_papers,
                    generate_paper_summary,
                    summarize_latest_author_paper,
                    scrape_papers,
                    save_papers_to_db,
                    publish_linkedin_post,
                    publish_paper_to_linkedin,
                    search_and_publish_paper,
                    create_linkedin_post_from_paper,
                ],
            )

        tools = pocket.get_tools()

        # Use the appropriate API key for the agent
        agent_api_key = os.getenv("OPENAI_API_KEY_AGENT")
        if not agent_api_key:
            raise ValueError("OPENAI_API_KEY_AGENT environment variable not set")
            
        llm = ChatOpenAI(model="gpt-4o", api_key=agent_api_key)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                (
                    "system",
                    """You are an advanced AI research assistant specialized in academic paper management, recommendation, and summarization.
                    You can help users manage research labs, track papers from lab members, recommend relevant papers, and generate summaries.
                    
                    Key capabilities:
                    - Create and manage research lab collections with member information
                    - Crawl Google Scholar pages of lab members to collect their papers
                    - Check for new papers by lab members
                    - Recommend relevant papers from arXiv based on lab research interests
                    - Generate comprehensive paper summaries that include context from related research
                    - Publish paper summaries to LinkedIn with PDF attachments
                    
                    When referring to labs, always use lowercase for the lab_name parameter.
                    User names should not contain special characters or spaces.
                    """,
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_summary_length=1000,  # Limit summary length to control token usage
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        cls.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )

        return cls

    @classmethod
    def terminal_mode(cls, executor: AgentExecutor | None = None) -> None:
        """Run the agent in terminal mode for interactive user input.
        
        This method provides an interactive command-line interface for the agent.
        Users can input commands and receive responses from the agent. The interface
        supports various operations including lab management, paper tracking, and
        LinkedIn publishing.
        
        Args:
            executor (AgentExecutor, optional): A pre-configured agent executor.
                If None, the class's executor will be used.
        """
        executor = executor or cls.executor
        print("\n\n\n")
        print("Hello, this is the LinkedIn Publishing Agent.")
        print("You can create lab collections, track papers from lab members, get recommendations, and generate summaries.")
        while True:
            print("user(q to quit) : ", end="")
            user_input = input()
            if user_input == "q":
                print("Good bye!")
                break

            # Check if lab name is mentioned to automatically check for new papers
            if "lab" in user_input.lower():
                # Extract potential lab names from the input
                words = user_input.lower().split()
                for i, word in enumerate(words):
                    if word in ["lab", "laboratory"] and i > 0:
                        potential_lab = words[i - 1]
                        # Try to get lab info to validate it exists
                        try:
                            lab_info = get_lab_info(potential_lab)
                            if isinstance(lab_info, dict):  # Lab exists
                                print(f"Checking for new papers for {potential_lab} lab...")
                                # Don't actually call it here, let the agent handle it if appropriate
                        except:
                            pass

            response = executor.invoke({"input": user_input})
            print()
