from environment import EmailEnv

def run_medium_task(agent):

    env = EmailEnv()

    total_score = 0
    n = 5

    for _ in range(n):

        observation = env.reset()

        action = agent(observation.email_text)

        score = 0

        if action.category == env.current_email["category"]:
            score += 0.5

        if action.priority == env.current_email["priority"]:
            score += 0.5

        total_score += score

    return total_score / n