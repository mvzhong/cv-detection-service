from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import detections

app = FastAPI()
api_v1 = FastAPI(title="API V1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1.include_router(detections.router)

app.mount("/api/v1", api_v1)


@app.get("/")
def root():
    return {"message": "Hello World"}
