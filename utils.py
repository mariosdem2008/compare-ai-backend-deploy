def compare_workouts(workout_a, workout_b):
    try:
        a_distance = workout_a.get("data", {}).get("distance", 0)
        b_distance = workout_b.get("data", {}).get("distance", 0)

        a_avg_hr = workout_a.get("data", {}).get("avg_hr", 0)
        b_avg_hr = workout_b.get("data", {}).get("avg_hr", 0)

        comparison = {
            "distance_diff": round(abs(a_distance - b_distance), 2),
            "avg_hr_diff": abs(a_avg_hr - b_avg_hr),
            "who_ran_more": "A" if a_distance > b_distance else "B" if b_distance > a_distance else "Equal",
            "hr_comment": (
                "Workout A had a higher avg HR"
                if a_avg_hr > b_avg_hr
                else "Workout B had a higher avg HR"
                if b_avg_hr > a_avg_hr
                else "Both had the same avg HR"
            ),
        }

        return comparison
    except Exception as e:
        return {"error": str(e)}
