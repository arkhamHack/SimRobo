#!/usr/bin/env python3
import os
import subprocess
import argparse
import time
import webbrowser
import signal
import sys
import uvicorn
import multiprocessing
import asyncio

def run_backend(gui=True, port=8080):
    """Run backend server using uvicorn"""
    env = os.environ.copy()
    env["ENABLE_GUI"] = str(gui).lower()
    command = f"uvicorn backend.server:socket_app --host 0.0.0.0 --port {port} --ws websockets --reload"
    print(f"Starting backend with command: {command}")
    return subprocess.Popen(command, shell=True, env=env)

def run_react_frontend(port=3000):
    """Run React frontend server using npm start (optional helper function)"""
    # This function is provided for convenience, but typically you'd run
    # the React frontend separately with 'npm start' in the frontend directory
    os.chdir('frontend')
    command = "npm start"
    return subprocess.Popen(command, shell=True)

def signal_handler(_, __):
    """Signal handler for clean exit"""
    print('Exiting gracefully...')
    sys.exit(0)

async def run_backend_directly(gui=True, port=8080):
    """Run backend server in-process using uvicorn"""
    os.environ["ENABLE_GUI"] = str(gui).lower()
    config = uvicorn.Config("backend.server:socket_app", host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    await server.serve()

def start_backend_process(gui=True, port=8080):
    """Start backend server in a separate process"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_backend_directly(gui=gui, port=port))

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="V-JEPA 2 Robot Control with FastAPI and React")
    parser.add_argument('--gui', action='store_true', help='Enable MuJoCo GUI')
    parser.add_argument('--with-react', action='store_true', help='Also start React frontend (normally you would start React separately)')
    parser.add_argument('--backend-port', type=int, default=8080, help='Backend server port')
    parser.add_argument('--react-port', type=int, default=3000, help='React frontend server port')
    parser.add_argument('--open-browser', action='store_true', help='Open browser automatically')
    parser.add_argument('--direct', action='store_true', help='Run backend directly in-process instead of as subprocess')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend server
        print(f"Starting backend server on port {args.backend_port}...")
        if args.direct:
            # Run backend in a separate process but with direct control
            backend_process = multiprocessing.Process(
                target=start_backend_process,
                args=(args.gui, args.backend_port)
            )
            backend_process.start()
        else:
            # Run backend as a separate subprocess
            backend_process = run_backend(gui=args.gui, port=args.backend_port)
        
        # Start React frontend if requested
        if args.with_react:
            print(f"Starting React frontend server on port {args.react_port}...")
            frontend_process = run_react_frontend(port=args.react_port)
            
            if args.open_browser:
                print("Opening browser...")
                time.sleep(2)  # Give React a bit more time to start
                webbrowser.open(f"http://localhost:{args.react_port}")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    
    finally:
        # Clean up backend process
        if backend_process:
            if isinstance(backend_process, multiprocessing.Process):
                backend_process.terminate()
                backend_process.join()
                print("Backend server stopped")
            else:
                backend_process.terminate()
                backend_process.wait()
            print("Backend server stopped")
            
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
            print("React frontend server stopped")

if __name__ == "__main__":
    main()
