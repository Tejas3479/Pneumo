def generate_risk_explanation(shipment, predicted_delay, risk_level, feature_importance_dict, feature_names):
    # Sort features by importance values
    top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join([f"{name} (weight: {val:.2f})" for name, val in top_features])
    explanation = (
        f"Shipment {shipment['ShipmentID']} from {shipment['Origin']} to {shipment['Destination']} "
        f"has a predicted delay of {predicted_delay:.1f} days ({risk_level} risk). "
        f"The top contributing risk factors are: {top_str}. "
        f"It is recommended to evaluate alternative routing or carrier assignments."
    )
    return explanation

def classify_risk(delay_days):
    if delay_days < 1:
        return "Low"
    elif delay_days < 3:
        return "Medium"
    else:
        return "High"
