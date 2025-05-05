import os
import inspect
from hyperpocket.util.function_to_model import function_to_model
from hyperpocket.tool.function.tool import FunctionTool

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
from tools.rednote_publisher import (publish_rednote_post, publish_paper_to_rednote,
                                    search_and_publish_paper_rednote, create_rednote_content_with_gpt)
from tools.paper_scraper import scrape_papers, save_papers_to_db, create_linkedin_post_from_paper

def agent(pocket: PocketLangchain):
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
                """You are an advanced AI research assistant specialized in academic paper management, recommendation, and publishing.
                You can help users manage research labs, track papers from lab members, recommend relevant papers, generate summaries,
                and publish content to LinkedIn and Xiaohongshu (Rednote).
                
                Key capabilities:
                - Create and manage research lab collections with member information
                - Crawl Google Scholar pages of lab members to collect their papers
                - Check for new papers by lab members
                - Recommend relevant papers from arXiv based on lab research interests
                - Generate comprehensive paper summaries that include context from related research
                - Publish paper summaries to LinkedIn with PDF attachments
                - Publish paper summaries to Xiaohongshu (Rednote) with PDF attachments
                
                When referring to labs, always use lowercase for the lab_name parameter.
                User names should not contain special characters or spaces.
                
                For publishing, you can:
                - Post to LinkedIn using publish_linkedin_post
                - Post to Rednote using publish_rednote_post
                - Publish a paper summary to LinkedIn using publish_paper_to_linkedin
                - Publish a paper summary to Rednote using publish_paper_to_rednote
                - Search and publish papers to either platform using search_and_publish_paper or search_and_publish_paper_rednote
                - Generate Rednote content using create_rednote_content_with_gpt
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
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("\n\n\n")
    print("Hello, this is the Unified Research Assistant.")
    print("You can manage research labs, track papers, get recommendations, generate summaries,")
    print("and publish content to LinkedIn and Xiaohongshu (Rednote).")
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

        response = agent_executor.invoke({"input": user_input})
        print()


if __name__ == "__main__":
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
            publish_rednote_post,
            publish_paper_to_rednote,
            search_and_publish_paper_rednote,
            create_rednote_content_with_gpt,
        ]
    
    print("🔍 Checking tools for compatibility...\n")
    for tool in tools:
        # unwrap if it's a FunctionTool
        if isinstance(tool, FunctionTool):
            fn = tool.func
            name = tool.name
        else:
            fn = tool
            name = getattr(tool, "__name__", repr(tool))

        try:
            # This will raise if the signature or annotations are invalid
            function_to_model(fn).model_json_schema()
        except Exception as e:
            print(f"❌ Tool `{name}` failed to convert:")
            print(f"   → {type(e).__name__}: {e}\n")
            
    with PocketLangchain(
        tools=tools,
    ) as pocket:
        agent(pocket)