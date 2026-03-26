from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from build_vectorstore import build_vectorstore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/rebuildFaiss")
async def rebuild():
    return build_vectorstore()