from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

@app.get("/")
async def serve_html():
    # Vérifie que interface.html est bien dans le même dossier que test_app.py
    path = Path(__file__).parent / "interface.html"
    return FileResponse(path, media_type="text/html")

