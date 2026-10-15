import argparse
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the local DAG Lab workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--data-dir", help="Location for datasets and experiments")
    args = parser.parse_args()
    from .api import create_app
    app = create_app(args.data_dir)
    if not args.no_browser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
