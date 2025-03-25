import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools.chromadb import create_collection, add_pdf_documents, query_collection
from tools.sqlite import insert_user_data, get_user_data
from hyperdock_fileio import initialize_dock as fileio_dock
from hyperpocket.tool import from_dock

from hyperpocket_langchain import PocketLangchain


def agent(pocket: PocketLangchain):
    tools = pocket.get_tools()

    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            (
                "system",
                "You are a tool calling assistant. You can help the user by calling proper tools \
                User name should be in the format with out any special characters or spaces",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("\n\n\n")
    print("Hello, this is langchain agent.")
    while True:
        print("user(q to quit) : ", end="")
        user_input = input()
        if user_input == "q":
            print("Good bye!")
            break

        response = agent_executor.invoke({"input": user_input})
        print("langchain agent : ", response["output"])
        print()


if __name__ == "__main__":
    with PocketLangchain(
        tools=[
            create_collection,
            insert_user_data,
            get_user_data,
            *from_dock(fileio_dock()),
            add_pdf_documents,
            query_collection,
        ],
    ) as pocket:
        agent(pocket)
