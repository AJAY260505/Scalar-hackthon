from environment import EmailEnv


def run_hard_task(agent):

    env = EmailEnv()

    observation = env.reset()

    action = agent(observation.email_text)

    score = 0

    if action.category == env.current_email["category"]:
        score += 0.33

    if action.priority == env.current_email["priority"]:
        score += 0.33

    if action.action == env.current_email["action"]:
        score += 0.34

    return score