from os import path

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        app_dir=path.abspath(path.join(path.dirname(__file__), "..")),
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
    )
