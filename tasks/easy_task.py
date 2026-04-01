from environment import EmailEnv

def run_easy_task(agent):

    env = EmailEnv()

    total_score = 0
    n = 5

    for _ in range(n):

        observation = env.reset()

        action = agent(observation.email_text)

        if action.category == env.current_email["category"]:
            total_score += 1

    return total_score / n