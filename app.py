from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
from datetime import datetime

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
class Split(BaseModel):
    label: Optional[str]
    pace: Optional[str]
    km: Optional[str]
    split: Optional[int]
    time: Optional[str]

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
    splits: Optional[List[Dict[str, Any]]] = None
    time: Optional[str] = None  # total duration in "HH:MM:SS" format
    workout_date: Optional[str] = None

class CompareRequest(BaseModel):
    workout_a: Workout
    workout_b: Workout

# ---------- Utilities ----------
def pace_in_seconds(pace_str: str) -> int:
    if not pace_str or pace_str == "-- /km":
        return 0
    try:
        pace_str = pace_str.replace(':', "'").replace('"', '').strip()
        match = re.match(r"(\d+)'(\d+)", pace_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
    except:
        pass
    return 0

def _pace_in_minutes(pace_str: str) -> float:
    seconds = pace_in_seconds(pace_str)
    return seconds / 60 if seconds else 0.0

def time_in_seconds(time_str: str) -> int:
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    if not time_str:
        return 0
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        elif len(parts) == 1:  # SS
            return parts[0]
    except:
        pass
    return 0

def detect_session_type(splits) -> str:
    if not splits:
        return "continuous"
    
    has_rest = any(
        split.get("label", "").lower() == "rest" or 
        (split.get("pace") and _pace_in_minutes(split.get("pace")) > 10)
        for split in splits
    )
    return "intervals" if has_rest else "continuous"

def classify_workout(workout: Workout) -> str:
    splits = workout.splits or []
    avg_hr = workout.avg_hr or 0
    distance = workout.distance or 0
    pace = _pace_in_minutes(workout.pace)
    structure = detect_session_type(splits)
    duration = time_in_seconds(workout.time) if workout.time else 0

    if structure == "intervals":
        interval_distances = []
        for split in splits:
            label = split.get("label", "").lower()
            km_str = split.get("km", "0 km")
            try:
                km = float(km_str.split()[0])
            except (ValueError, IndexError):
                continue
            
            # Include only runs under 0.5 km to classify short intervals reliably
            if label == "run" and km <= 0.5:
                interval_distances.append(km)
            # Optionally skip "lap" splits or include them only if they're short enough
            elif label == "lap" and km <= 0.5:
                interval_distances.append(km)
            # You may also add other logic for 'interval' labels if applicable

        avg_interval_distance = np.mean(interval_distances) if interval_distances else 0

        if avg_interval_distance <= 0.5:
            return "short intervals (200-400m)"
        elif avg_interval_distance <= 1.0:
            return "mid-length intervals (800-1000m)"
        else:
            return "long intervals (mile+)"
    # Continuous runs
    if distance <= 5 and avg_hr > 180 and pace < 4.0:
        return "5K race or time trial"
    elif distance > 10 and avg_hr < 150:
        return "long easy run"
    elif 5 < distance <= 10 and 150 <= avg_hr <= 170:
        return "tempo run"
    elif distance > 21 and distance <= 42 and avg_hr < 150:
        return "marathon pace run"
    elif distance > 15 and 140 <= avg_hr <= 160:
        return "steady state run"
    else:
        return "general aerobic run"

# ---------- Insight Engine ----------
class InsightEngine:
    def __init__(self, a: Dict[str, Any], b: Dict[str, Any], pace_diff_sec: int):
        self.a = a
        self.b = b
        self.pace_diff_sec = pace_diff_sec
        self.insights = {
            "intensity": [], 
            "efficiency": [], 
            "strength": [], 
            "form": [], 
            "training_type": [], 
            "context": [],
            "physiological": [],
            "performance": []
        }
        self.key_takeaways = []
        self.recommendations = []
        self.risk_flags = []
        self.performance_trend = {}
        self.comparison_metrics = {}

    def analyze(self):
        self._workout_classification()
        self._relative_intensity()
        self._heart_rate()
        self._pace_analysis()
        self._cadence()
        self._elevation()
        self._effort()
        self._power()
        self._stride()
        self._quality()
        self._distance_duration()
        self._environment_context()
        self._efficiency_metrics()
        self._training_load()
        self._build_takeaways()
        self._generate_recommendations()
        self._evaluate_risks()
        self._evaluate_trends()
        return self._export()

    def _workout_classification(self):
        a_type = classify_workout(Workout(**self.a))
        b_type = classify_workout(Workout(**self.b))
        
        self.insights["training_type"].append(f"Workout A: {a_type}")
        self.insights["training_type"].append(f"Workout B: {b_type}")
        
        if a_type != b_type:
            self.insights["context"].append(
                f"Comparing different workout types: {a_type} vs {b_type}. "
                "Consider that each serves different training purposes."
            )

    def _relative_intensity(self):
        a_duration = time_in_seconds(self.a.get("time", "0:00"))
        b_duration = time_in_seconds(self.b.get("time", "0:00"))
        
        # Calculate intensity score (simple version)
        a_intensity = (self.a.get("avg_hr", 0) or 0) * (a_duration / 3600)  # HR * hours
        b_intensity = (self.b.get("avg_hr", 0) or 0) * (b_duration / 3600)
        
        self.comparison_metrics["intensity_score"] = {
            "a": round(a_intensity, 1),
            "b": round(b_intensity, 1),
            "difference": round(a_intensity - b_intensity, 1)
        }
        
        if a_intensity > b_intensity * 1.2:
            self.insights["intensity"].append(
                "Workout A was significantly more demanding in overall training load"
            )
        elif b_intensity > a_intensity * 1.2:
            self.insights["intensity"].append(
                "Workout B was significantly more demanding in overall training load"
            )

    def _heart_rate(self):
        avg_hr_diff = (self.a.get("avg_hr", 0) or 0) - (self.b.get("avg_hr", 0) or 0)
        max_hr_diff = (self.a.get("max_hr", 0) or 0) - (self.b.get("max_hr", 0) or 0)
        
        self.comparison_metrics["heart_rate"] = {
            "avg_hr_diff": avg_hr_diff,
            "max_hr_diff": max_hr_diff
        }
        
        if self.a.get("avg_hr") and self.b.get("avg_hr"):
            if self.a["avg_hr"] < self.b["avg_hr"]:
                hr_percent_diff = ((self.b["avg_hr"] - self.a["avg_hr"]) / self.a["avg_hr"]) * 100
                self.insights["physiological"].append(
                    f"Workout B had {abs(hr_percent_diff):.1f}% higher average HR ({self.b['avg_hr']} vs {self.a['avg_hr']}), "
                    "indicating significantly higher cardiovascular stress"
                )
            else:
                hr_percent_diff = ((self.a["avg_hr"] - self.b["avg_hr"]) / self.b["avg_hr"]) * 100
                self.insights["physiological"].append(
                    f"Workout A had {hr_percent_diff:.1f}% higher average HR ({self.a['avg_hr']} vs {self.b['avg_hr']})"
                )

        if self.a.get("max_hr") and self.b.get("max_hr"):
            if self.a["max_hr"] < self.b["max_hr"]:
                self.insights["intensity"].append(
                    f"Workout B reached higher max HR ({self.b['max_hr']} vs {self.a['max_hr']}), "
                    "suggesting more maximal effort"
                )
            elif self.a["max_hr"] > self.b["max_hr"]:
                self.insights["intensity"].append(
                    f"Workout A reached higher max HR ({self.a['max_hr']} vs {self.b['max_hr']})"
                )

    def _pace_analysis(self):
        a_splits = self.a.get("splits", [])
        b_splits = self.b.get("splits", [])
        
        # Basic pace comparison
        if self.pace_diff_sec < 0:
            self.insights["performance"].append(
                f"Workout B was faster by {-self.pace_diff_sec} sec/km — "
                "consider workout types when interpreting this difference"
            )
        elif self.pace_diff_sec > 0:
            self.insights["performance"].append(
                f"Workout A was faster by {self.pace_diff_sec} sec/km"
            )
        
        # Analyze pace consistency from splits
        if a_splits and b_splits:
            a_run_paces = [pace_in_seconds(split["pace"]) for split in a_splits 
                          if split.get("label", "").lower() in ["run", "lap"] and split.get("pace")]
            b_run_paces = [pace_in_seconds(split["pace"]) for split in b_splits 
                          if split.get("label", "").lower() in ["run", "lap"] and split.get("pace")]
            
            if a_run_paces and b_run_paces:
                a_pace_std = np.std(a_run_paces)
                b_pace_std = np.std(b_run_paces)
                
                self.comparison_metrics["pace_consistency"] = {
                    "a_std": round(a_pace_std, 1),
                    "b_std": round(b_pace_std, 1)
                }
                
                if a_pace_std > b_pace_std * 2:
                    self.insights["performance"].append(
                        "Workout A had more variable pacing (typical for interval sessions)"
                    )
                elif b_pace_std > a_pace_std * 2:
                    self.insights["performance"].append(
                        "Workout B had more variable pacing"
                    )
                
                # Detect negative/positive splits
                if len(a_run_paces) >= 3:
                    first_half = np.mean(a_run_paces[:len(a_run_paces)//2])
                    second_half = np.mean(a_run_paces[len(a_run_paces)//2:])
                    if second_half < first_half * 0.95:  # 5% faster
                        self.insights["performance"].append(
                            "Workout A showed negative splits (faster as it progressed)"
                        )
                
                if len(b_run_paces) >= 3:
                    first_half = np.mean(b_run_paces[:len(b_run_paces)//2])
                    second_half = np.mean(b_run_paces[len(b_run_paces)//2:])
                    if second_half < first_half * 0.95:  # 5% faster
                        self.insights["performance"].append(
                            "Workout B showed negative splits (faster as it progressed)"
                        )

    def _cadence(self):
        diff = (self.a.get("cadence_avg", 0) or 0) - (self.b.get("cadence_avg", 0) or 0)
        
        self.comparison_metrics["cadence"] = {
            "a_avg": self.a.get("cadence_avg"),
            "b_avg": self.b.get("cadence_avg"),
            "difference": diff
        }
        
        if abs(diff) <= 5:
            self.insights["form"].append(
                "Cadence was stable between workouts, suggesting consistent form"
            )
        elif diff > 5:
            self.insights["form"].append(
                f"Workout A had higher cadence (+{diff} spm), indicating quicker turnover"
            )
            if self.a.get("stride_length_avg") and self.b.get("stride_length_avg"):
                if self.a["stride_length_avg"] > self.b["stride_length_avg"]:
                    self.insights["form"].append(
                        "Despite higher cadence, Workout A also had longer stride length - "
                        "this combination suggests powerful running form"
                    )
        else:
            self.insights["form"].append(
                f"Workout B had higher cadence ({abs(diff)} spm more than Workout A)"
            )

    def _elevation(self):
        gain_diff = (self.a.get("elevation_gain", 0) or 0) - (self.b.get("elevation_gain", 0) or 0)
        avg_diff = (self.a.get("elevation_avg", 0) or 0) - (self.b.get("elevation_avg", 0) or 0)
        
        self.comparison_metrics["elevation"] = {
            "gain_diff": gain_diff,
            "avg_diff": avg_diff
        }
        
        if gain_diff > 10:
            self.insights["strength"].append(
                f"Workout A had {gain_diff}m more elevation gain, "
                "suggesting greater muscular strength demand"
            )
        elif gain_diff < -10:
            self.insights["strength"].append(
                f"Workout B had {abs(gain_diff)}m more elevation gain"
            )
        
        if abs(avg_diff) > 20:
            self.insights["context"].append(
                f"Significant elevation difference: Workout A at {self.a.get('elevation_avg')}m avg vs "
                f"Workout B at {self.b.get('elevation_avg')}m avg"
            )

    def _effort(self):
        a_score = self.a.get("effort_score", 0) or 0
        b_score = self.b.get("effort_score", 0) or 0
        diff = a_score - b_score
        
        if a_score and b_score:
            if diff > 0:
                self.insights["intensity"].append(
                    f"Workout A had higher effort score ({a_score} vs {b_score})"
                )
            elif diff < 0:
                self.insights["intensity"].append(
                    f"Workout B had higher effort score ({b_score} vs {a_score})"
                )

    def _power(self):
        avg_diff = (self.a.get("running_power_avg", 0) or 0) - (self.b.get("running_power_avg", 0) or 0)
        max_diff = (self.a.get("running_power_max", 0) or 0) - (self.b.get("running_power_max", 0) or 0)
        
        self.comparison_metrics["power"] = {
            "avg_diff": avg_diff,
            "max_diff": max_diff
        }
        
        if abs(avg_diff) > 10:
            if avg_diff > 0:
                self.insights["performance"].append(
                    f"Workout A had higher average running power (+{avg_diff}W)"
                )
            else:
                self.insights["performance"].append(
                    f"Workout B had higher average running power ({abs(avg_diff)}W more)"
                )
        
        if abs(max_diff) > 20:
            if max_diff > 0:
                self.insights["performance"].append(
                    f"Workout A reached higher peak power (+{max_diff}W)"
                )
            else:
                self.insights["performance"].append(
                    f"Workout B reached higher peak power ({abs(max_diff)}W more)"
                )

    def _stride(self):
        diff = (self.a.get("stride_length_avg", 0) or 0) - (self.b.get("stride_length_avg", 0) or 0)
        
        if abs(diff) > 5:
            if diff > 0:
                self.insights["form"].append(
                    f"Workout A had longer stride length (+{diff}cm), "
                    "which could indicate more powerful push-off"
                )
            else:
                self.insights["form"].append(
                    f"Workout B had longer stride length ({abs(diff)}cm more)"
                )
        
        # Stride length to height ratio (approximate)
        if self.a.get("stride_length_avg") and self.b.get("stride_length_avg"):
            a_ratio = (self.a["stride_length_avg"] / 100) / 1.7  # Assuming ~1.7m height
            b_ratio = (self.b["stride_length_avg"] / 100) / 1.7
            if a_ratio > 1.0 or b_ratio > 1.0:
                self.insights["form"].append(
                    "One or both workouts showed stride lengths exceeding body height, "
                    "which may indicate overstriding"
                )

    def _quality(self):
        if self.a.get("is_quality_session") and self.b.get("is_quality_session"):
            if self.a["is_quality_session"] and not self.b["is_quality_session"]:
                self.insights["training_type"].append(
                    "Workout A was marked as a quality session while Workout B was not"
                )
            elif not self.a["is_quality_session"] and self.b["is_quality_session"]:
                self.insights["training_type"].append(
                    "Workout B was marked as a quality session while Workout A was not"
                )

    def _distance_duration(self):
        dist_diff = (self.a.get("distance", 0) or 0) - (self.b.get("distance", 0) or 0)
        a_duration = time_in_seconds(self.a.get("time", "0:00"))
        b_duration = time_in_seconds(self.b.get("time", "0:00"))
        duration_diff = a_duration - b_duration
        
        self.comparison_metrics["distance_duration"] = {
            "distance_diff": dist_diff,
            "duration_diff": duration_diff
        }
        
        if dist_diff > 0:
            self.insights["strength"].append(
                f"Workout A covered {dist_diff:.2f}km more distance"
            )
        elif dist_diff < 0:
            self.insights["strength"].append(
                f"Workout B covered {abs(dist_diff):.2f}km more distance"
            )
        
        if duration_diff > 0:
            hours, remainder = divmod(abs(duration_diff), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.insights["context"].append(
                f"Workout A was {hours}h {minutes}m {seconds}s longer in duration"
            )
        elif duration_diff < 0:
            hours, remainder = divmod(abs(duration_diff), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.insights["context"].append(
                f"Workout B was {hours}h {minutes}m {seconds}s longer in duration"
            )

    def _environment_context(self):
        humidity_a = self.a.get("humidity")
        temperature_a = self.a.get("temperature")
        wind_a = self.a.get("wind_speed")
        
        humidity_b = self.b.get("humidity")
        temperature_b = self.b.get("temperature")
        wind_b = self.b.get("wind_speed")
        
        env_notes = []
        
        if humidity_b is not None and humidity_b > 75:
            env_notes.append("high humidity")
        if temperature_b is not None and temperature_b > 28:
            env_notes.append("elevated temperature")
        if wind_b is not None and wind_b > 15:
            env_notes.append("strong wind")
        
        if env_notes:
            self.insights["context"].append(
                f"Workout B had challenging conditions: {', '.join(env_notes)}"
            )

    def _efficiency_metrics(self):
        """Calculate combined efficiency metrics"""
        # Running Efficiency Index (simplified)
        if self.a.get("running_power_avg") and self.a.get("pace"):
            a_pace_sec = pace_in_seconds(self.a["pace"])
            a_efficiency = (self.a["running_power_avg"] / a_pace_sec) if a_pace_sec else 0
        
        if self.b.get("running_power_avg") and self.b.get("pace"):
            b_pace_sec = pace_in_seconds(self.b["pace"])
            b_efficiency = (self.b["running_power_avg"] / b_pace_sec) if b_pace_sec else 0
        
        if a_efficiency and b_efficiency:
            self.comparison_metrics["efficiency_index"] = {
                "a": round(a_efficiency, 2),
                "b": round(b_efficiency, 2),
                "difference": round(a_efficiency - b_efficiency, 2)
            }
            
            if a_efficiency > b_efficiency * 1.1:
                self.insights["efficiency"].append(
                    f"Workout A showed {((a_efficiency - b_efficiency)/b_efficiency)*100:.1f}% better running efficiency "
                    "(more speed per watt of power)"
                )
            elif b_efficiency > a_efficiency * 1.1:
                self.insights["efficiency"].append(
                    f"Workout B showed {((b_efficiency - a_efficiency)/a_efficiency)*100:.1f}% better running efficiency"
                )

    def _training_load(self):
        """Estimate training load using TRIMP method"""
        a_duration = time_in_seconds(self.a.get("time", "0:00")) / 3600  # hours
        b_duration = time_in_seconds(self.b.get("time", "0:00")) / 3600
        
        # Simple TRIMP calculation: duration * avg_hr
        a_trimp = a_duration * (self.a.get("avg_hr", 0) or 0)
        b_trimp = b_duration * (self.b.get("avg_hr", 0) or 0)
        
        self.comparison_metrics["training_load"] = {
            "a": round(a_trimp, 1),
            "b": round(b_trimp, 1),
            "difference": round(a_trimp - b_trimp, 1),
            "percent_change": ((a_trimp - b_trimp) / b_trimp) * 100 if b_trimp else 0
        }
        
        if a_trimp > b_trimp * 1.5:
            self.insights["physiological"].append(
                f"Workout A had {((a_trimp - b_trimp)/b_trimp)*100:.1f}% greater training load "
                "(combination of duration and intensity)"
            )
        elif b_trimp > a_trimp * 1.5:
            self.insights["physiological"].append(
                f"Workout B had {((b_trimp - a_trimp)/a_trimp)*100:.1f}% greater training load"
            )

    def _build_takeaways(self):
        # Prioritize the most significant insights
        priority_order = [
            "performance", "physiological", "intensity", 
            "efficiency", "training_type", "form", 
            "strength", "context"
        ]
        
        for category in priority_order:
            if category in self.insights and self.insights[category]:
                self.key_takeaways.extend(self.insights[category][:2])  # Take max 2 per category
        
        # Ensure we have at least 3 takeaways
        if len(self.key_takeaways) < 3:
            remaining = sum(len(v) for v in self.insights.values()) - len(self.key_takeaways)
            if remaining > 0:
                for category in priority_order:
                    if category in self.insights:
                        for insight in self.insights[category]:
                            if insight not in self.key_takeaways:
                                self.key_takeaways.append(insight)
                                if len(self.key_takeaways) >= 3:
                                    break
                        if len(self.key_takeaways) >= 3:
                            break
        
        self.key_takeaways = self.key_takeaways[:5]  # Limit to top 5

    def _generate_recommendations(self):
        # Based on heart rate
        if self.b.get("max_hr") and self.b["max_hr"] > 190:
            self.recommendations.append(
                "After high-intensity sessions like Workout B, ensure 48 hours recovery "
                "before another hard effort"
            )
        
        # Based on cadence
        if self.a.get("cadence_avg") and self.a["cadence_avg"] < 170:
            self.recommendations.append(
                "Consider cadence drills to improve running economy for workouts like Workout A"
            )
        
        # Based on workout type
        a_type = classify_workout(Workout(**self.a))
        b_type = classify_workout(Workout(**self.b))
        
        if "interval" in a_type and "interval" in b_type:
            self.recommendations.append(
                "You're doing multiple interval sessions - ensure adequate easy runs "
                "between for recovery"
            )
        
        # Based on training load
        if self.comparison_metrics.get("training_load", {}).get("percent_change", 0) > 30:
            self.recommendations.append(
                "Significant increase in training load detected - monitor for signs of overtraining"
            )

    def _evaluate_risks(self):
        if self.a.get("avg_hr") and self.a["avg_hr"] > 175:
            self.risk_flags.append(
                "High average HR in Workout A - ensure proper recovery was taken"
            )
        
        if self.b.get("avg_hr") and self.b["avg_hr"] > 175:
            self.risk_flags.append(
                "High average HR in Workout B - this intensity should be limited to 1-2x weekly"
            )
        
        if (self.a.get("running_power_max") and self.b.get("running_power_max") and 
            abs(self.a["running_power_max"] - self.b["running_power_max"]) > 50):
            self.risk_flags.append(
                "Large variation in peak power output - check for consistent effort or possible data issues"
            )

    def _evaluate_trends(self):
        self.performance_trend = {
            "aerobic_efficiency": "improving" if (
                self.a.get("avg_hr") and self.b.get("avg_hr") and 
                self.a["avg_hr"] < self.b["avg_hr"] and 
                pace_in_seconds(self.a.get("pace", "0'0")) <= pace_in_seconds(self.b.get("pace", "0'0"))
            ) else "declining",

            "speed_endurance": "better" if self.pace_diff_sec > 0 else "worse",

            "cadence_consistency": "stable" if abs(
                (self.a.get("cadence_avg", 0) or 0) - (self.b.get("cadence_avg", 0) or 0)
            ) <= 5 else "changing",

            "power_output": "higher" if (
                self.a.get("running_power_avg") and self.b.get("running_power_avg") and 
                self.a["running_power_avg"] > self.b["running_power_avg"]
            ) else "lower"
        }


    def _export(self):
        return {
            "categorized_insights": self.insights,
            "key_takeaways": self.key_takeaways,
            "recommendations": self.recommendations,
            "risk_flags": self.risk_flags,
            "performance_trend": self.performance_trend,
            "comparison_metrics": self.comparison_metrics,
            "summary": self._generate_summary(),
            "workout_types": {
                "a": classify_workout(Workout(**self.a)),
                "b": classify_workout(Workout(**self.b))
            }
        }

    def _generate_summary(self):
        if not self.key_takeaways:
            return "See detailed insights for comparison results"
        
        summary = []
        a_type = classify_workout(Workout(**self.a))
        b_type = classify_workout(Workout(**self.b))
        
        if a_type != b_type:
            summary.append(f"Comparing {a_type} with {b_type}:")
        
        summary.extend(self.key_takeaways[:3])
        
        if self.recommendations:
            summary.append("Recommendations: " + self.recommendations[0])
        
        return " ".join(summary)

# ---------- Comparison Logic ----------
def compare_workouts(a: Workout, b: Workout):
    a_dict, b_dict = a.dict(), b.dict()
    pace_diff = pace_in_seconds(a_dict.get("pace", "0'0")) - pace_in_seconds(b_dict.get("pace", "0'0"))
    engine = InsightEngine(a_dict, b_dict, pace_diff)
    insights = engine.analyze()
def _generate_detailed_summary(self) -> str:
    a = self.a
    b = self.b

    # Classify workouts
    a_type = classify_workout(Workout(**a))
    b_type = classify_workout(Workout(**b))

    # Calculate key diffs
    pace_diff_sec = self.pace_diff_sec
    pace_diff_str = f"{abs(pace_diff_sec)} sec/km {'faster' if pace_diff_sec > 0 else 'slower'}"

    cadence_diff = (a.get("cadence_avg") or 0) - (b.get("cadence_avg") or 0)
    stride_diff = (a.get("stride_length_avg") or 0) - (b.get("stride_length_avg") or 0)
    elev_gain_diff = (a.get("elevation_gain") or 0) - (b.get("elevation_gain") or 0)

    eff_a = (a.get("running_power_avg") or 0) / pace_in_seconds(a.get("pace")) if (a.get("running_power_avg") and a.get("pace")) else 0
    eff_b = (b.get("running_power_avg") or 0) / pace_in_seconds(b.get("pace")) if (b.get("running_power_avg") and b.get("pace")) else 0
    efficiency_diff_pct = ((eff_b - eff_a) / eff_a * 100) if (eff_a > 0) else 0

    # Executive summary
    summary = []
    summary.append(f"🏁 Executive Summary")
    summary.append(f"Workout A outperformed in speed and intensity, clocking a {abs(pace_diff_sec)} sec/km {'faster' if pace_diff_sec > 0 else 'slower'} pace while hitting a higher max HR ({a.get('max_hr', 'N/A')} vs. {b.get('max_hr', 'N/A')}) — a sign of pushing limits during intervals.")

    if efficiency_diff_pct > 5:
        summary.append(f"Despite this, Workout B showed {efficiency_diff_pct:.1f}% better running efficiency, thanks to its higher cadence ({b.get('cadence_avg', 0)} vs. {a.get('cadence_avg', 0)} spm) and more consistent form.")
    else:
        summary.append(f"Efficiency between the workouts was comparable.")

    # Key contrasts
    summary.append(f"\nKey Contrasts:")
    summary.append(f"Workout A: Explosive intervals with longer strides (+{stride_diff}cm) and {int(elev_gain_diff)}m more elevation gain, demanding muscular strength.")
    summary.append(f"Workout B: Smoother cadence and better efficiency, but slower pace—likely due to focus on form or fatigue management.")

    # Takeaway
    summary.append(f"\nTakeaway: You’re balancing raw speed (A) with efficiency (B). The trade-off suggests room to refine cadence in Workout A while maintaining power.")

    # Key Performance Metrics table (simple textual version)
    summary.append(f"\n📊 Key Performance Metrics")
    summary.append(f"{'Metric':<20}{'Workout A':<12}{'Workout B':<12}{'Difference'}")
    summary.append(f"{'-'*60}")
    def fmt_diff(val, rev=False):
        # Show positive difference as (B > A) or (A > B), reversed if rev=True
        if val == 0:
            return "Equal"
        elif val > 0:
            return "A > B" if not rev else "B > A"
        else:
            return "B > A" if not rev else "A > B"

    summary.append(f"{'Distance (km)':<20}{a.get('distance', 0):<12.2f}{b.get('distance', 0):<12.2f}{fmt_diff(b.get('distance',0) - a.get('distance',0), rev=True)}")
    summary.append(f"{'Avg HR (bpm)':<20}{a.get('avg_hr', 0):<12}{b.get('avg_hr', 0):<12}{(a.get('avg_hr',0)-b.get('avg_hr',0)):+} bpm")
    summary.append(f"{'Max HR (bpm)':<20}{a.get('max_hr', 0):<12}{b.get('max_hr', 0):<12}{(a.get('max_hr',0)-b.get('max_hr',0)):+} bpm")
    summary.append(f"{'Pace (min/km)':<20}{a.get('pace', 'N/A'):<12}{b.get('pace', 'N/A'):<12}{pace_diff_str}")
    summary.append(f"{'Cadence (spm)':<20}{a.get('cadence_avg', 0):<12}{b.get('cadence_avg', 0):<12}{cadence_diff:+} spm")
    summary.append(f"{'Elevation Gain (m)':<20}{int(a.get('elevation_gain', 0)):<12}{int(b.get('elevation_gain', 0)):<12}{int(elev_gain_diff):+} m")
    summary.append(f"{'Running Power (avg)':<20}{a.get('running_power_avg', 0):<12}{b.get('running_power_avg', 0):<12}{(a.get('running_power_avg',0) - b.get('running_power_avg', 0)):+} W")

    # Insight Highlights
    summary.append(f"\n🔍 Insight Highlights")
    summary.append(f"🔸 Intensity vs. Efficiency")
    if a.get('avg_hr') and b.get('avg_hr'):
        summary.append(f"Workout A’s higher HR and pace reflect high-intensity efforts, ideal for speed development.")
        summary.append(f"Workout B’s lower HR + higher cadence suggests better economy—likely due to smoother mechanics.")

    summary.append(f"🔸 Strength & Form")
    summary.append(f"Workout A’s elevation gain ({int(a.get('elevation_gain', 0))}m) and longer strides indicate power-focused intervals.")
    summary.append(f"Workout B’s consistent cadence hints at repetition practice (e.g., pacing control).")

    summary.append(f"🔸 Training Type")
    summary.append(f"Both workouts were {a_type}, but A emphasized power, while B prioritized efficiency.")

    # Flagged Considerations
    summary.append(f"\n⚠️ Flagged Considerations")
    if a.get('cadence_avg', 0) < 120:
        summary.append(f"Cadence Gap: Workout A’s low cadence ({a.get('cadence_avg')} spm) risks overstriding. Drills could help.")
    if elev_gain_diff > 100:
        summary.append(f"Elevation Impact: Workout A’s hills explain its higher HR/power—great for strength but taxing.")
    summary.append(f"Recovery Needed: Two interval sessions close together? Ensure easy runs between hard efforts.")

    # Performance Trends
    summary.append(f"\n📈 Performance Trends")
    summary.append(f"Aerobic Efficiency: {'Declining (Higher HR in Workout A)' if self.performance_trend.get('aerobic_efficiency') == 'declining' else 'Improving'}")
    summary.append(f"Speed Endurance: {'Better (Workout A’s pace dominance)' if self.performance_trend.get('speed_endurance') == 'better' else 'Worse'}")
    summary.append(f"Cadence Consistency: {self.performance_trend.get('cadence_consistency', 'N/A').capitalize()}")
    summary.append(f"Power Output: {self.performance_trend.get('power_output', 'N/A').capitalize()}")

    # Recommended Next Actions
    summary.append(f"\n✅ Recommended Next Actions")
    summary.append(f"Cadence Drills: Add strides or metronome runs to boost Workout A’s turnover without losing power.")
    summary.append(f"Recovery Balance: Follow these sessions with low-HR runs to avoid overtraining.")
    summary.append(f"Hill vs. Flat Intervals: Rotate hill repeats (A) and flat-speed intervals (B) to blend strength/speed.")

    # Coach’s Takeaway
    summary.append(f"\n🧠 Coach’s Takeaway")
    summary.append(f'“Workout A proves you can push hard on hills and speed, but Workout B shows your efficient side. To progress, bridge the gap: Keep A’s power while adopting B’s cadence. And don’t forget—intervals are taxing! Slot in recovery days to let these gains stick.”')

    return "\n".join(summary)


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)