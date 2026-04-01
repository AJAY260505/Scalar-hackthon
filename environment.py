import json
import random
from pydantic import BaseModel


class Observation(BaseModel):
    email_text: str
    email_id: int


class Action(BaseModel):
    category: str
    priority: str
    action: str


class Reward(BaseModel):
    score: float


class EmailEnv:

    def __init__(self):

        with open("dataset/emails.json") as f:
            self.data = json.load(f)

        self.current_email = None


    def reset(self):

        self.current_email = random.choice(self.data)

        return Observation(
            email_text=self.current_email["text"],
            email_id=self.current_email["id"]
        )


    def step(self, action: Action):

        score = 0

        # category is most important
        if action.category == self.current_email["category"]:
            score += 0.4
        else:
            score -= 0.05   # small penalty


        # priority importance
        if action.priority == self.current_email["priority"]:
            score += 0.3


        # action correctness
        if action.action == self.current_email["action"]:
            score += 0.3


        # keep score between 0 and 1
        score = max(0, min(score, 1))


        done = True


        return {
            "observation": None,
            "reward": score,
            "done": done,
            "info": {}
        }


    def state(self):

        return self.current_email