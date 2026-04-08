from environment import EmailEnv

def run_medium_task(agent):
    """
    Medium: Classify category + priority
    Scores must be strictly between 0.0 and 1.0 (exclusive)
    """
    env = EmailEnv()
    total_score = 0
    n = 10
    
    for _ in range(n):
        observation = env.reset()
        action = agent(observation.email_text)
        
        score = 0
        
        # Category (0.5 weight)
        if action.category == env.current_email["category"]:
            score += 0.5
        
        # Priority (0.5 weight)
        if action.priority == env.current_email["priority"]:
            score += 0.5
        
        total_score += score
    
    # Convert to average score
    base_score = total_score / n
    
    # Add epsilon to shift away from boundaries
    epsilon = 0.001
    score = base_score * (1 - 2 * epsilon) + epsilon
    
    # Clamp to ensure strictly between 0 and 1
    score = min(max(score, 0.01), 0.99)
    
    return round(score, 2)
