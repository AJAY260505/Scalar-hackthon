from environment import Action
from tasks.easy_task import run_easy_task
from tasks.medium_task import run_medium_task
from tasks.hard_task import run_hard_task


def simple_agent(email_text):

    text = email_text.lower()

    # billing related emails
    if any(word in text for word in [
        "refund","charged","invoice","payment",
        "transaction","price","billing","amount"
    ]):
        return Action(
            category="billing",
            priority="high",
            action="escalate"
        )


    # technical issues
    if any(word in text for word in [
        "password","error","bug","crash",
        "login","server","issue","problem"
    ]):
        return Action(
            category="technical",
            priority="high",
            action="escalate"
        )


    # account related
    if any(word in text for word in [
        "account","profile","delete",
        "remove","update","change"
    ]):
        return Action(
            category="account",
            priority="medium",
            action="reply"
        )


    # feedback or appreciation emails
    if any(word in text for word in [
        "thank","great","love","good",
        "awesome","nice","excellent"
    ]):
        return Action(
            category="general",
            priority="low",
            action="archive"
        )


    # default fallback
    return Action(
        category="general",
        priority="medium",
        action="reply"
    )


# run evaluation tasks

easy_score = run_easy_task(simple_agent)
medium_score = run_medium_task(simple_agent)
hard_score = run_hard_task(simple_agent)


# structured logging format required by OpenEnv validator

print("\n[START]")

print("[STEP] task=easy score=", easy_score)

print("[STEP] task=medium score=", medium_score)

print("[STEP] task=hard score=", hard_score)

print("[END]")