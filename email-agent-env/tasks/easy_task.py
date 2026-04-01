from environment import EmailEnv, Action


def run_easy_task(agent):

    env = EmailEnv()

    observation = env.reset()

    action = agent(observation.email_text)

    score = 1.0 if action.category == env.current_email["category"] else 0.0

    return score