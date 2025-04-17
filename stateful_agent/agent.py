import os

from model import StatefulAgentExecutor
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools.chromadb import create_collection, add_pdf_documents, query_collection
from tools.sqlite import insert_user_data, get_user_data
from hyperdock_fileio import initialize_dock as fileio_dock
from hyperpocket.tool import from_dock

from hyperpocket_langchain import PocketLangchain

class SimpleStatefulAgent(StatefulAgentExecutor):
    """A simple stateful agent implementation using LangChain and various tools."""
    executor: AgentExecutor

    @classmethod
    def create_agent(cls):
        """Create and configure a new agent with tools and memory."""
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

        # Using ConversationSummaryMemory instead of ConversationBufferMemory
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_summary_length=1000  # Limit summary length to control token usage
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
    def terminal_mode(cls, executor: AgentExecutor | None = None):
        """Run the agent in terminal mode for interactive user input."""
        executor = executor or cls.executor
        print("\n\n\n")
        print("Hello, this is langchain agent.")
        while True:
            print("user(q to quit) : ", end="")
            user_input = input()
            if user_input == "q":
                print("Good bye!")
                break

            response = executor.invoke({"input": user_input})
            print("langchain agent : ", response["output"])
            print()

if __name__ == "__main__":
    agent_executor = SimpleStatefulAgent.create_agent()
    agent_executor.terminal_mode()  # Uses class executor
    # Or alternatively:
    # SimpleStatefulAgent.terminal_mode(agent_executor.executor)  # Uses provided executor
