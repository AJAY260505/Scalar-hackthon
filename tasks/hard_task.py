from environment import EmailEnv

def run_hard_task(agent):
    """
    Hard: Classify category + priority + action
    Scores must be strictly between 0.0 and 1.0 (exclusive)
    """
    env = EmailEnv()
    total_score = 0
    n = 10
    
    for _ in range(n):
        observation = env.reset()
        action = agent(observation.email_text)
        
        score = 0
        
        # Category (0.33 weight)
        if action.category == env.current_email["category"]:
            score += 0.33
        
        # Priority (0.33 weight)
        if action.priority == env.current_email["priority"]:
            score += 0.33
        
        # Action (0.34 weight)
        if action.action == env.current_email["action"]:
            score += 0.34
        
        total_score += score
    
    # Convert to average score
    base_score = total_score / n
    
    # Add epsilon to shift away from boundaries
    epsilon = 0.001
    score = base_score * (1 - 2 * epsilon) + epsilon
    
    # Clamp to ensure strictly between 0 and 1
    score = min(max(score, 0.01), 0.99)
    
    return round(score, 2)
