import os
import sys
import argparse
import importlib
import inspect
import click
import pathlib
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import FileResponse
import uvicorn

current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent

template_dir = project_root / 'front_end' / 'templates'
static_dir = project_root / 'front_end' / 'static'

print(template_dir, static_dir)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(template_dir))

def load_subclass_from_file(file_path, base_class_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    module_name = "temp_agent_module"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base_class = None
    for obj_name, obj in module.__dict__.items():
        if inspect.isclass(obj) and obj.__name__ == base_class_name:
            base_class = obj
            break

    if base_class is None:
        raise ImportError(f"Base class '{base_class_name}' not found in {file_path}")

    for obj_name, obj in module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class:
            return obj

    raise ImportError(f"No subclass of '{base_class_name}' found in {file_path}")

# Global agent instance
agent_instance = None

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", context= {"request": request})

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get('message')
    session_id = data.get('session_id')
    
    if not message:
        return JSONResponse(content={'error': 'No message provided'}, status_code=400)
    
    try:
        response = agent_instance.invoke({"input": message})
        return JSONResponse(content={'response': response['output']})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.get("/api/files")
async def get_files():
    try:
        # Get list of files from the data directory
        data_dir = os.getenv('DEFAULT_DATA_DIR', './data')
        files = []
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return JSONResponse(content={'files': files})
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@click.command()
@click.option('--file', type=str, required=True, help='Path to the agent file')
@click.option('--mode', type=str, default='terminal', help='Deployment mode: web or terminal')
def deploy_agent(file, mode):
    if mode not in ['web', 'terminal']:
        raise ValueError("Invalid mode. Please use 'web' or 'terminal'.")
    
    global agent_instance, app
    AgentClass = load_subclass_from_file(file, "StatefulAgentExecutor")
    agent_instance = AgentClass.create_agent().executor
    
    if mode == 'web':
        host = '0.0.0.0'
        port = 6001
        print(f"Starting web server on {host}:{port}")
        print(f"Template directory: {template_dir}")
        print(f"Static directory: {static_dir}")
        uvicorn.run(app, host=host, port=port)
    else:
        agent_instance.terminal_mode()
