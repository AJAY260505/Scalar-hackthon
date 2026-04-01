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

        self.index = 0
        self.current_email = None

    def reset(self):
        self.index = 0
        self.current_email = random.choice(self.data)

        return Observation(
            email_text=self.current_email["text"],
            email_id=self.current_email["id"]
        )

    def step(self, action: Action):

        score = 0

        if action.category == self.current_email["category"]:
            score += 0.33

        if action.priority == self.current_email["priority"]:
            score += 0.33

        if action.action == self.current_email["action"]:
            score += 0.34

        done = True

        return {
            "observation": None,
            "reward": score,
            "done": done,
            "info": {}
        }

    def state(self):
        return self.current_email