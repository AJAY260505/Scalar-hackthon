from environment import EmailEnv

def run_easy_task(agent):
    """
    Easy: Classify email category only
    Scores must be strictly between 0.0 and 1.0 (exclusive)
    """
    env = EmailEnv()
    correct = 0
    total = 10
    
    for _ in range(total):
        observation = env.reset()
        action = agent(observation.email_text)
        
        if action.category == env.current_email["category"]:
            correct += 1
    
    # Convert to score: correct/total
    base_score = correct / total
    
    # Add epsilon (0.001) to shift away from boundaries
    # This ensures we never return exactly 0.0 or 1.0
    epsilon = 0.001
    score = base_score * (1 - 2 * epsilon) + epsilon
    
    # Clamp to ensure strictly between 0 and 1
    score = min(max(score, 0.01), 0.99)
    
    return round(score, 2)
