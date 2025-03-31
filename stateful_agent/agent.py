import os
import argparse
from flask import Flask, render_template, request, jsonify
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools.chromadb import create_collection, add_pdf_documents, query_collection
from tools.sqlite import insert_user_data, get_user_data
from hyperdock_fileio import initialize_dock as fileio_dock
from hyperpocket.tool import from_dock
from hyperpocket_langchain import PocketLangchain

def create_agent():
    """Create a new agent instance with tools and memory."""
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
        
        llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = ChatPromptTemplate.from_messages([
            ("placeholder", "{chat_history}"),
            (
                "system",
                "You are a tool calling assistant. You can help the user by calling proper tools. \
                User name should be in the format with out any special characters or spaces",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent_executor

def run_cli_mode():
    """Run the agent in CLI mode."""
    agent_executor = create_agent()
    print("\nWelcome to the Stateful Agent CLI!")
    print("Type 'q' to quit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'q':
            print("Goodbye!")
            break
            
        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nAgent:", response['output'], "\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")

def run_web_mode():
    """Run the agent in web mode."""
    app = Flask(__name__, 
                template_folder='front_end/templates',
                static_folder='front_end/static')

    # Global variables to store agent instances
    agent_instances = {}

    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')

    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Handle chat messages from the frontend."""
        data = request.json
        user_input = data.get('message')
        session_id = data.get('session_id')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Create or get agent instance for this session
        if session_id not in agent_instances:
            agent_instances[session_id] = create_agent()
        
        try:
            response = agent_instances[session_id].invoke({"input": user_input})
            return jsonify({
                'response': response['output'],
                'session_id': session_id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/files', methods=['GET'])
    def get_files():
        """Get list of generated files."""
        # TODO: Implement file listing logic
        return jsonify({'files': []})

    app.run(debug=True, port=5000)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Stateful Agent - Choose your interface mode')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli',
                      help='Choose the interface mode: cli (default) or web')
    args = parser.parse_args()

    if args.mode == 'web':
        print("Starting Stateful Agent in web mode...")
        print("Access the interface at http://localhost:5000")
        run_web_mode()
    else:
        print("Starting Stateful Agent in CLI mode...")
        run_cli_mode()

if __name__ == "__main__":
    main()
