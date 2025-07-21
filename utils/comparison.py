def compare_documents(doc_a, doc_b):
    data_a = extract_data(doc_a)
    data_b = extract_data(doc_b)

    comparison = {
        "Distance (km)": data_a["distance"] - data_b["distance"],
        "Average Pace": f"{data_a['pace']} vs {data_b['pace']}",
        "Avg HR": data_a["avg_hr"] - data_b["avg_hr"],
        "Max HR": data_a["max_hr"] - data_b["max_hr"],
        "Cadence Avg": data_a["cadence_avg"] - data_b["cadence_avg"],
        "Stride Length Avg": data_a["stride_length_avg"] - data_b["stride_length_avg"],
    }

    summary = format_summary(comparison)

    return {
        "comparison": comparison,
        "summary": summary
    }


def extract_data(doc):
    # Try both completed and stored structures
    data = doc.get("data", {}) if "data" in doc else doc

    return {
        "distance": float(data.get("distance", 0.0)),
        "pace": data.get("pace", "N/A"),
        "avg_hr": int(data.get("avg_hr", 0)),
        "max_hr": int(data.get("max_hr", 0)),
        "cadence_avg": int(data.get("cadence_avg", 0)),
        "stride_length_avg": int(data.get("stride_length_avg", 0)),
    }


def format_summary(diff):
    lines = []
    for key, value in diff.items():
        if isinstance(value, (int, float)):
            lines.append(f"{key}: {value:+.2f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
