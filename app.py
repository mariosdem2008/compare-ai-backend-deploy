# Updated version of app.py with all requested enhancements

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data Models ----------
class Workout(BaseModel):
    avg_hr: Optional[int] = None
    max_hr: Optional[int] = None
    pace: str  # format "mm'ss" or "mm:ss"
    distance: float
    cadence_avg: Optional[int] = None
    cadence_max: Optional[int] = None
    elevation_gain: Optional[float] = None
    elevation_loss: Optional[float] = None
    elevation_avg: Optional[float] = None
    elevation_max: Optional[float] = None
    elevation_min: Optional[float] = None
    running_power_avg: Optional[int] = None
    running_power_max: Optional[int] = None
    stride_length_avg: Optional[int] = None
    effort_score: Optional[int] = None
    is_quality_session: Optional[bool] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

class CompareRequest(BaseModel):
    workout_a: Workout
    workout_b: Workout

# ---------- Utilities ----------
def pace_to_seconds(pace_str: str) -> int:
    if not pace_str:
        return 0
    try:
        pace_str = pace_str.replace(':', "'").replace('"', '').strip()
        match = re.match(r"(\d+)'(\d+)", pace_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
    except:
        pass
    return 0

def classify_session(workout: Workout) -> str:
    pace_sec = pace_to_seconds(workout.pace)
    if workout.is_quality_session:
        if workout.distance >= 10 and pace_sec < 300:
            return "tempo"
        elif workout.distance < 5:
            return "interval"
        else:
            return "threshold"
    return "easy"

# ---------- Insight Engine ----------
class InsightEngine:
    def __init__(self, a: Dict[str, Any], b: Dict[str, Any], pace_diff_sec: int):
        self.a = a
        self.b = b
        self.pace_diff_sec = pace_diff_sec
        self.insights = {
            "intensity": [], "efficiency": [], "strength": [], "form": [], "training_type": [], "context": []
        }
        self.key_takeaways = []
        self.recommendations = []
        self.risk_flags = []
        self.performance_trend = {}

    def analyze(self):
        self._heart_rate()
        self._pace()
        self._cadence()
        self._elevation()
        self._effort()
        self._power()
        self._stride()
        self._quality()
        self._distance()
        self._environment_context()
        self._build_takeaways()
        self._generate_recommendations()
        self._evaluate_risks()
        self._evaluate_trends()
        return self._export()

    def _environment_context(self):
        if (self.a.get("temperature") and self.b.get("temperature")) and self.b["temperature"] > self.a["temperature"]:
            self.insights["context"].append("Workout B was done in hotter conditions — improved pace may suggest stronger thermoregulation.")
        if self.b.get("humidity", 0) > 75:
            self.insights["context"].append("Workout B occurred in high humidity — performance may have been impacted.")

    def _heart_rate(self):
        if self.a["avg_hr"] and self.b["avg_hr"]:
            if self.a["avg_hr"] < self.b["avg_hr"]:
                self.insights["intensity"].append("Workout A had lower average heart rate, showing better aerobic efficiency or reduced strain.")
            elif self.a["avg_hr"] > self.b["avg_hr"]:
                self.insights["intensity"].append("Workout B had lower average heart rate, possibly indicating easier intensity or aerobic gains.")
            else:
                self.insights["intensity"].append("Average heart rates were similar, indicating matched effort levels.")

        if self.a["max_hr"] and self.b["max_hr"]:
            if self.a["max_hr"] < self.b["max_hr"]:
                self.insights["intensity"].append("Workout A peaked at a lower max heart rate, possibly due to controlled pacing.")
            elif self.a["max_hr"] > self.b["max_hr"]:
                self.insights["intensity"].append("Workout B peaked at a lower max heart rate, reflecting lower maximum effort.")
            else:
                self.insights["intensity"].append("Both workouts reached similar peak heart rates.")

    def _pace(self):
        if self.pace_diff_sec < 0:
            self.insights["efficiency"].append(f"Workout B was faster by {-self.pace_diff_sec} sec/km — a sign of improved pace or intensity.")
        elif self.pace_diff_sec > 0:
            self.insights["efficiency"].append(f"Workout A was faster by {self.pace_diff_sec} sec/km — good speed potential.")
        else:
            self.insights["efficiency"].append("Both workouts had identical paces.")

    def _cadence(self):
        diff = (self.a.get("cadence_avg", 0) or 0) - (self.b.get("cadence_avg", 0) or 0)
        if abs(diff) <= 5:
            self.insights["form"].append("Cadence was stable, suggesting consistent form.")
        elif diff > 5:
            self.insights["form"].append("Workout A had higher cadence, indicating quicker turnover and possibly improved running economy.")
        else:
            self.insights["form"].append("Workout B had higher cadence, pointing to efficient stride mechanics.")

    def _elevation(self):
        diff = (self.a.get("elevation_gain", 0) or 0) - (self.b.get("elevation_gain", 0) or 0)
        if diff > 10:
            self.insights["strength"].append("Workout A involved more elevation, suggesting muscular strength or hill work.")
        elif diff < -10:
            self.insights["strength"].append("Workout B had more elevation gain, challenging strength and stamina.")
        else:
            self.insights["strength"].append("Elevation gain was similar — terrain likely comparable.")

    def _effort(self):
        diff = (self.a.get("effort_score", 0) or 0) - (self.b.get("effort_score", 0) or 0)
        if diff > 0:
            self.insights["intensity"].append("Workout A had a higher effort score, likely a more demanding session.")
        elif diff < 0:
            self.insights["intensity"].append("Workout B had a higher effort score, possibly from greater distance or effort.")
        else:
            self.insights["intensity"].append("Effort scores were the same — sessions were similarly taxing.")

    def _power(self):
        diff = (self.a.get("running_power_avg", 0) or 0) - (self.b.get("running_power_avg", 0) or 0)
        if diff > 10:
            self.insights["efficiency"].append("Workout A showed greater average running power — an indicator of better output.")
        elif diff < -10:
            self.insights["efficiency"].append("Workout B had more average running power, pointing to stronger force or pace.")
        else:
            self.insights["efficiency"].append("Running power was stable across both workouts.")

    def _stride(self):
        diff = (self.a.get("stride_length_avg", 0) or 0) - (self.b.get("stride_length_avg", 0) or 0)
        if diff > 5:
            self.insights["form"].append("Workout A had a longer stride length, which could boost speed but monitor for overstriding.")
        elif diff < -5:
            self.insights["form"].append("Workout B had longer strides, potentially improving efficiency.")
        else:
            self.insights["form"].append("Stride lengths were consistent.")

    def _quality(self):
        self.insights["training_type"].append(f"Workout A was classified as {classify_session(Workout(**self.a))} session.")
        self.insights["training_type"].append(f"Workout B was classified as {classify_session(Workout(**self.b))} session.")

    def _distance(self):
        if self.a["distance"] > self.b["distance"]:
            self.insights["strength"].append("Workout A covered more distance, showing endurance capability.")
        elif self.a["distance"] < self.b["distance"]:
            self.insights["strength"].append("Workout B covered more distance, reflecting longer aerobic work.")
        else:
            self.insights["strength"].append("Distances were equal.")

    def _build_takeaways(self):
        all_insights = sum(self.insights.values(), [])
        self.key_takeaways = all_insights[:3]

    def _generate_recommendations(self):
        if self.pace_diff_sec > 20:
            self.recommendations.append("Include speed intervals to sustain improved pace.")
        if self.a.get("stride_length_avg") and self.a["stride_length_avg"] < 100:
            self.recommendations.append("Consider drills to improve stride length for Workout A.")

    def _evaluate_risks(self):
        if self.a.get("avg_hr") and self.a["avg_hr"] > 175:
            self.risk_flags.append("High average HR in Workout A — monitor recovery.")
        if self.b.get("avg_hr") and self.b["avg_hr"] > 175:
            self.risk_flags.append("High average HR in Workout B — risk of overtraining.")

    def _evaluate_trends(self):
        self.performance_trend = {
            "aerobic_efficiency": "improving" if self.a["avg_hr"] < self.b["avg_hr"] else "declining",
            "speed_endurance": "better" if self.pace_diff_sec > 0 else "worse",
            "cadence": "stable" if abs((self.a.get("cadence_avg") or 0) - (self.b.get("cadence_avg") or 0)) <= 5 else "changing"
        }

    def _export(self):
        return {
            "categorized_insights": self.insights,
            "key_takeaways": self.key_takeaways,
            "recommendations": self.recommendations,
            "risk_flags": self.risk_flags,
            "performance_trend": self.performance_trend,
            "summary": self._generate_summary()
        }

    def _generate_summary(self):
        return " ".join(self.key_takeaways) if len(self.key_takeaways) >= 3 else "See categorized insights."

# ---------- Comparison Logic ----------
def compare_workouts(a: Workout, b: Workout):
    a_dict, b_dict = a.dict(), b.dict()
    pace_diff = pace_to_seconds(a_dict.get("pace", "0'0")) - pace_to_seconds(b_dict.get("pace", "0'0"))
    engine = InsightEngine(a_dict, b_dict, pace_diff)
    insights = engine.analyze()

    return {
        "distance_diff": round((a_dict.get("distance") or 0) - (b_dict.get("distance") or 0), 2),
        "avg_hr_diff": (a_dict.get("avg_hr") or 0) - (b_dict.get("avg_hr") or 0),
        "max_hr_diff": (a_dict.get("max_hr") or 0) - (b_dict.get("max_hr") or 0),
        "cadence_avg_diff": (a_dict.get("cadence_avg") or 0) - (b_dict.get("cadence_avg") or 0),
        "cadence_max_diff": (a_dict.get("cadence_max") or 0) - (b_dict.get("cadence_max") or 0),
        "elevation_gain_diff": (a_dict.get("elevation_gain") or 0) - (b_dict.get("elevation_gain") or 0),
        "effort_score_diff": (a_dict.get("effort_score") or 0) - (b_dict.get("effort_score") or 0),
        "running_power_avg_diff": (a_dict.get("running_power_avg") or 0) - (b_dict.get("running_power_avg") or 0),
        "stride_length_avg_diff": (a_dict.get("stride_length_avg") or 0) - (b_dict.get("stride_length_avg") or 0),
        "pace_diff_seconds_per_km": pace_diff,
        "who_ran_more": "Workout A" if a_dict["distance"] > b_dict["distance"] else ("Workout B" if a_dict["distance"] < b_dict["distance"] else "Equal"),
        **insights
    }

# ---------- Endpoint ----------
@app.post("/compare")
async def compare(data: CompareRequest):
    result = compare_workouts(data.workout_a, data.workout_b)
    logger.info(f"Comparison result: {result}")
    return {"result": result}