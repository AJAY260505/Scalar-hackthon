from environment import EmailEnv

def run_medium_task(agent):
    """
    Medium: Classify category + priority
    Scores must be strictly between 0.0 and 1.0
    """
    env = EmailEnv()
    total_score = 0
    n = 10  # Increased from 5 to 10 for better score distribution
    
    for _ in range(n):
        observation = env.reset()
        action = agent(observation.email_text)
        
        score = 0
        
        # Category (0.5 weight)
        if action.category == env.current_email["category"]:
            score += 0.5
        else:
            score += 0.05
        
        # Priority (0.5 weight)
        if action.priority == env.current_email["priority"]:
            score += 0.5
        else:
            score += 0.05
        
        total_score += score
    
    # Clamp to ensure strictly between 0 and 1 (never exactly 0 or 1)
    avg_score = total_score / n
    avg_score = min(max(avg_score, 0.01), 0.99)  # Clamp to [0.01, 0.99]
    
    return round(avg_score, 2)