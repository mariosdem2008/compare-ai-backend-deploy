from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import logging
from utils import compare_workouts

app = FastAPI()

# Enable CORS (you can limit to your Flutter app domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)

class CompareRequest(BaseModel):
    workout_a: Dict[str, Any]
    workout_b: Dict[str, Any]

@app.post("/compare")
async def compare_endpoint(data: CompareRequest):
    logging.info("=== Workout A ===")
    logging.info(data.workout_a)
    logging.info("=== Workout B ===")
    logging.info(data.workout_b)

    result = compare_workouts(data.workout_a, data.workout_b)

    logging.info("=== Comparison Result ===")
    logging.info(result)

    return {"result": result}
