import os
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Get the directory where this file is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Mount static directory
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(base_dir, "index.html"))

@app.get("/{path:path}")
async def serve(path: str):
    if os.path.exists(os.path.join(base_dir, "static", path)):
        return FileResponse(os.path.join(base_dir, "static", path))
    else:
        return FileResponse(os.path.join(base_dir, "index.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontend server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
