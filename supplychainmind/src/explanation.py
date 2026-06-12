def generate_risk_explanation(shipment_row, predicted_delay, confidence=None) -> str:
    """
    Generates a natural language risk explanation for a shipment based on predicted delay and features.
    """
    shipment_id = shipment_row.get("shipment_id", "N/A")
    origin = shipment_row.get("origin", "Unknown")
    destination = shipment_row.get("destination", "Unknown")
    carrier = shipment_row.get("carrier", "Unknown")
    
    # Simple risk percentage heuristic: delay of 10 days = 100% risk, etc.
    risk = int(min(100, max(0, (predicted_delay / 10.0) * 100)))
    days = round(predicted_delay, 1)
    
    # Identify top risk factors based on thresholds
    risk_factors = []
    if shipment_row.get("weather_risk", 0) > 0.4:
        risk_factors.append("severe weather")
    if shipment_row.get("congestion", 0) > 0.4:
        risk_factors.append("port congestion")
    if shipment_row.get("geopolitical_risk", 0) > 0.3:
        risk_factors.append("geopolitical risk")
    if shipment_row.get("supplier_health", 100) < 80:
        risk_factors.append("low supplier financial health")
        
    if not risk_factors:
        risk_factors.append("normal transit variance")
        
    list_of_top_factors = ", ".join(risk_factors)
    
    # Select recommended action
    if days >= 3.0:
        if "severe weather" in risk_factors:
            action = "Reroute shipment to avoid weather system or delay departure."
        elif "port congestion" in risk_factors:
            action = "Use secondary port or negotiate fast-track clearance."
        elif "geopolitical risk" in risk_factors:
            action = "Consult trade compliance and prepare alternative routing."
        else:
            action = "Consider switching to express carrier tier."
    else:
        action = "Monitor shipment status via standard tracking dashboard."
        
    return (
        f"Shipment {shipment_id} from {origin} to {destination} via {carrier} "
        f"has a {risk}% chance of being delayed by {days} days. "
        f"The main risk factors are {list_of_top_factors}. "
        f"Recommended action: {action}"
    )

