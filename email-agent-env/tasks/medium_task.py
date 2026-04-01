from environment import EmailEnv


def run_medium_task(agent):

    env = EmailEnv()

    observation = env.reset()

    action = agent(observation.email_text)

    score = 0

    if action.category == env.current_email["category"]:
        score += 0.5

    if action.priority == env.current_email["priority"]:
        score += 0.5

    return score