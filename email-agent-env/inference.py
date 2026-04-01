from environment import Action
from tasks.easy_task import run_easy_task
from tasks.medium_task import run_medium_task
from tasks.hard_task import run_hard_task


def simple_agent(email_text):

    email_text = email_text.lower()

    if "refund" in email_text or "charged" in email_text:
        return Action(
            category="billing",
            priority="high",
            action="escalate"
        )

    if "password" in email_text:
        return Action(
            category="technical",
            priority="medium",
            action="reply"
        )

    if "cancel" in email_text:
        return Action(
            category="account",
            priority="high",
            action="escalate"
        )

    if "thank" in email_text:
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


easy_score = run_easy_task(simple_agent)
medium_score = run_medium_task(simple_agent)
hard_score = run_hard_task(simple_agent)

print("\nRESULTS")
print("Easy Task Score:", easy_score)
print("Medium Task Score:", medium_score)
print("Hard Task Score:", hard_score)