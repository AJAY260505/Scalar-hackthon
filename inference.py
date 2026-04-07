import os
from environment import EmailEnv, Action
from tasks.easy_task import run_easy_task
from tasks.medium_task import run_medium_task
from tasks.hard_task import run_hard_task
import random

# Required environment variables - MUST use these
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "sk-default")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Initialize OpenAI client with provided credentials
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

Q_TABLE = {}
ACTION_SPACE = [
    {"category":"billing","priority":"high","action":"escalate"},
    {"category":"billing","priority":"medium","action":"reply"},
    {"category":"technical","priority":"high","action":"escalate"},
    {"category":"technical","priority":"medium","action":"reply"},
    {"category":"account","priority":"medium","action":"reply"},
    {"category":"account","priority":"high","action":"escalate"},
    {"category":"general","priority":"low","action":"archive"},
    {"category":"general","priority":"medium","action":"reply"},
]

LEARNING_RATE = 0.3
EXPLORATION_RATE = 0.3

def get_state(email_text):
    text = email_text.lower()
    
    if any(w in text for w in ["refund","charged","invoice","payment","transaction"]):
        return "billing"
    if any(w in text for w in ["password","error","bug","crash","login","server"]):
        return "technical"
    if any(w in text for w in ["account","profile","delete","remove","update"]):
        return "account"
    if any(w in text for w in ["thank","great","love","good","excellent"]):
        return "general"
    
    return "general"

def classify_with_llm(email_text):
    """
    Use OpenAI client to classify email via LLM.
    This ensures API_BASE_URL and API_KEY are used.
    """
    try:
        prompt = f"""Classify this email into ONE of these categories: billing, technical, account, or general.
Also assign priority (low, medium, high) and action (reply, escalate, archive).

Email: {email_text}

Respond in this exact format:
category: [billing|technical|account|general]
priority: [low|medium|high]
action: [reply|escalate|archive]"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an email classification expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        text = response.choices[0].message.content.strip()
        
        # Parse response
        lines = text.split('\n')
        result = {}
        for line in lines:
            if 'category:' in line.lower():
                result['category'] = line.split(':')[1].strip().lower()
            elif 'priority:' in line.lower():
                result['priority'] = line.split(':')[1].strip().lower()
            elif 'action:' in line.lower():
                result['action'] = line.split(':')[1].strip().lower()
        
        return result
    except Exception as e:
        # Fallback to rule-based if API fails
        print(f"[DEBUG] LLM classification failed: {e}", flush=True)
        return None

def choose_action(state):
    if random.random() < EXPLORATION_RATE:
        return random.choice(ACTION_SPACE)
    
    if state in Q_TABLE:
        best_action_key = max(Q_TABLE[state], key=Q_TABLE[state].get)
        return eval(best_action_key)
    
    return random.choice(ACTION_SPACE)

def update_q(state, action_dict, reward):
    action_key = str(action_dict)
    
    if state not in Q_TABLE:
        Q_TABLE[state] = {}
    
    if action_key not in Q_TABLE[state]:
        Q_TABLE[state][action_key] = 0
    
    old_value = Q_TABLE[state][action_key]
    new_value = old_value + LEARNING_RATE * (reward - old_value)
    Q_TABLE[state][action_key] = new_value

def train_agent(episodes=50):
    env = EmailEnv()
    
    for _ in range(episodes):
        obs = env.reset()
        state = get_state(obs.email_text)
        
        # Try LLM classification first (uses API)
        llm_result = classify_with_llm(obs.email_text)
        if llm_result and all(k in llm_result for k in ['category', 'priority', 'action']):
            action_dict = llm_result
        else:
            # Fallback to rule-based
            action_dict = choose_action(state)
        
        action = Action(**action_dict)
        result = env.step(action)
        reward = result["reward"]
        update_q(state, action_dict, reward)

def rl_agent(email_text):
    state = get_state(email_text)
    
    # Try LLM classification first (uses API)
    llm_result = classify_with_llm(email_text)
    if llm_result and all(k in llm_result for k in ['category', 'priority', 'action']):
        action_dict = llm_result
    else:
        # Fallback to Q-table or random
        if state in Q_TABLE:
            best_action_key = max(Q_TABLE[state], key=Q_TABLE[state].get)
            action_dict = eval(best_action_key)
        else:
            action_dict = random.choice(ACTION_SPACE)
    
    return Action(**action_dict)

if __name__ == "__main__":
    # Training phase
    train_agent()
    
    # Evaluation phase
    easy_score = run_easy_task(rl_agent)
    medium_score = run_medium_task(rl_agent)
    hard_score = run_hard_task(rl_agent)
    
    # EXACT OUTPUT FORMAT - Required structured logging
    print("[START]")
    print(f"[STEP] task=easy score={easy_score}")
    print(f"[STEP] task=medium score={medium_score}")
    print(f"[STEP] task=hard score={hard_score}")
    print("[END]")