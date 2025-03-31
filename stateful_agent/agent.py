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
from tools.linkedin_publisher import publish_linkedin_post


def agent(pocket: PocketLangchain):
    tools = pocket.get_tools()

    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

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
                
                When referring to labs, always use lowercase for the lab_name parameter.
                User names should not contain special characters or spaces.
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
    print("Hello, this is the Paper Recommendation & Summary Agent.")
    print(
        "You can create lab collections, track papers from lab members, get recommendations, and generate summaries."
    )
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
        # print("langchain agent : ", response["output"]) # Abundant output
        print()


if __name__ == "__main__":
    with PocketLangchain(
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
            publish_linkedin_post,
        ],
    ) as pocket:
        agent(pocket)
