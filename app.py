from fastapi import FastAPI
from environment import EmailEnv, Action
from tasks.easy_task import run_easy_task
from tasks.medium_task import run_medium_task
from tasks.hard_task import run_hard_task

app = FastAPI()

env = EmailEnv()


def simple_agent(email_text):

    text = email_text.lower()

    if any(word in text for word in [
        "refund","charged","invoice","payment",
        "transaction","price","billing","amount"
    ]):
        return Action(
            category="billing",
            priority="high",
            action="escalate"
        )

    if any(word in text for word in [
        "password","error","bug","crash",
        "login","server","issue","problem"
    ]):
        return Action(
            category="technical",
            priority="high",
            action="escalate"
        )

    if any(word in text for word in [
        "account","profile","delete",
        "remove","update","change"
    ]):
        return Action(
            category="account",
            priority="medium",
            action="reply"
        )

    if any(word in text for word in [
        "thank","great","love","good"
    ]):
        return Action(
            category="general",
            priority="low",
            action="archive"
        )

    return Action(
        category="general",
        priority="medium",
        action="reply"
    )


@app.get("/")
def home():

    easy_score = run_easy_task(simple_agent)
    medium_score = run_medium_task(simple_agent)
    hard_score = run_hard_task(simple_agent)

    return {
        "easy_score": easy_score,
        "medium_score": medium_score,
        "hard_score": hard_score
    }


@app.get("/reset")
def reset_env():

    observation = env.reset()

    return {
        "email": observation.email_text,
        "id": observation.email_id
    }