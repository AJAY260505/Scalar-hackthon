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
        return Action(category="billing", priority="high", action="escalate")

    if any(word in text for word in [
        "password","error","bug","crash","login","server","issue"
    ]):
        return Action(category="technical", priority="high", action="escalate")

    if any(word in text for word in [
        "account","profile","delete","remove","update","change"
    ]):
        return Action(category="account", priority="medium", action="reply")

    if any(word in text for word in [
        "thank","great","love","good","excellent"
    ]):
        return Action(category="general", priority="low", action="archive")

    return Action(category="general", priority="medium", action="reply")


@app.get("/")
def home():
    """Health check - returns 200"""
    return {"status": "ready"}


@app.get("/inference")
def run_inference():
    """Run inference and return task scores"""
    try:
        easy_score = run_easy_task(simple_agent)
        medium_score = run_medium_task(simple_agent)
        hard_score = run_hard_task(simple_agent)
        
        return {
            "status": "success",
            "tasks": {
                "easy": easy_score,
                "medium": medium_score,
                "hard": hard_score
            },
            "output": {
                "START": True,
                "STEP": [
                    {"task": "easy", "score": easy_score},
                    {"task": "medium", "score": medium_score},
                    {"task": "hard", "score": hard_score}
                ],
                "END": True
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "tasks": {
                "easy": 0.0,
                "medium": 0.0,
                "hard": 0.0
            }
        }


@app.post("/reset")
def reset_env():
    """Reset environment and return observation"""
    observation = env.reset()
    return {
        "email_text": observation.email_text,
        "email_id": observation.email_id
    }


@app.post("/step")
def step_env(action: Action):
    """Take action and return reward"""
    result = env.step(action)
    return {
        "reward": result["reward"],
        "done": result["done"]
    }


def main():
    """Main entry point for the server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()