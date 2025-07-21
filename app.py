from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from Flutter

@app.route("/compare", methods=["POST"])
def compare_workouts():
    try:
        data = request.get_json()

        workout_a = data.get("workout_a", {}).get("data", {})
        workout_b = data.get("workout_b", {}).get("data", {})

        # Determine types
        type_a = detect_workout_type(workout_a)
        type_b = detect_workout_type(workout_b)

        summary = {
            "type_a": type_a,
            "type_b": type_b,
            "comparison": {},
        }

        # Extract values
        comparison = {
            "Distance (km)": extract_distance(workout_a) - extract_distance(workout_b),
            "Average Pace": extract_pace(workout_a) + " vs " + extract_pace(workout_b),
            "Avg HR": extract_value(workout_a, "avg_hr") - extract_value(workout_b, "avg_hr"),
            "Max HR": extract_value(workout_a, "max_hr") - extract_value(workout_b, "max_hr"),
            "Cadence Avg": extract_value(workout_a, "cadence_avg") - extract_value(workout_b, "cadence_avg"),
        }

        summary["comparison"] = comparison

        return jsonify({
            "status": "success",
            "summary": format_summary(comparison),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helpers

def detect_workout_type(doc):
    if "data" in doc and isinstance(doc["data"], dict):
        return "completed"
    if "phases" in doc:
        return "stored"
    return "unknown"

def extract_value(doc, key):
    return (
        doc.get(key)
        or doc.get("data", {}).get(key)
        or 0
    )

def extract_distance(doc):
    return float(
        doc.get("distance")
        or doc.get("data", {}).get("distance")
        or 0.0
    )

def extract_pace(doc):
    return (
        doc.get("pace")
        or doc.get("data", {}).get("pace")
        or "N/A"
    )

def format_summary(diff):
    lines = []
    for k, v in diff.items():
        if isinstance(v, (int, float)):
            lines.append(f"{k}: {v:+.2f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

if __name__ == "__main__":
    app.run(debug=True)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
