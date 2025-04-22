import os
import sys
import argparse
from flask import Flask, render_template, request, jsonify
from agent import agent, cli_mode
from hyperpocket_langchain import PocketLangchain
from tools.chromadb import create_collection, add_pdf_documents, query_collection
from tools.sqlite import insert_user_data, get_user_data
from hyperdock_fileio import initialize_dock as fileio_dock
from hyperpocket.tool import from_dock

current_dir = os.path.dirname(os.path.abspath(__file__))

template_dir = os.path.join(current_dir, 'front_end', 'templates')
static_dir = os.path.join(current_dir, 'front_end', 'static')

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

# Global agent instance
agent_instance = None

def create_agent():
    global agent_instance
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
        agent_instance = agent(pocket)
    return agent_instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    session_id = data.get('session_id')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = agent_instance.invoke({"input": message})
        return jsonify({'response': response['output']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files')
def get_files():
    try:
        # Get list of files from the data directory
        data_dir = os.getenv('DEFAULT_DATA_DIR', './data')
        files = []
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description='Deploy a stateful agent')
    parser.add_argument('--file', type=str, help='Path to the agent file')
    parser.add_argument('--mode', type=str, choices=['web', 'cli'], default='cli',
                      help='Deployment mode: web or cli (default: cli)')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        # Create agent instance
        create_agent()
        
        # 使用固定的端口 6001，避免与 Uvicorn 冲突
        host = '0.0.0.0'
        port = 6001
        
        print(f"Starting web server on {host}:{port}")
        print(f"Template directory: {template_dir}")
        print(f"Static directory: {static_dir}")
        app.run(host=host, port=port, debug=True)
    else:
        # CLI mode
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
            agent_executor = agent(pocket)
            cli_mode(agent_executor)

if __name__ == '__main__':
    main() 