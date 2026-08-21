from typing import Dict, Any

def extract_detection_metrics(results_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts core detection metrics from the dictionary returned by Ultralytics val().
    """
    results = {}

    if not results_dict:
        return results

    if 'metrics/precision(B)' in results_dict:
        results["precision"] = results_dict['metrics/precision(B)']
    if 'metrics/recall(B)' in results_dict:
        results["recall"] = results_dict['metrics/recall(B)']
    if 'metrics/mAP50(B)' in results_dict:
        results["mAP50"] = results_dict['metrics/mAP50(B)']
    if 'metrics/mAP50-95(B)' in results_dict:
        results["mAP50-95"] = results_dict['metrics/mAP50-95(B)']

    for k, v in results.items():
        if hasattr(v, "item"):
            results[k] = v.item()

    return results
