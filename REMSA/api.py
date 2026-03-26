from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from REMSA.agent_orchestrator import FMSAgent
from REMSA.benchmark_api import router as benchmark_router

app = FastAPI()
agent = FMSAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],        # must include OPTIONS
    allow_headers=["*"],
)

app.include_router(benchmark_router)

class Query(BaseModel):

    query: str
@app.options("/query")
def options_query():
    return JSONResponse(status_code=200, content={})

@app.post("/query")
def api_query(data: Query):
    result = agent.step(data.query)
    return result