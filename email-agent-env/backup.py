import os
from openai import OpenAI
from environment import EmailEnv, Action

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

env = EmailEnv()

observation = env.reset()

prompt = f"""
Classify this email.

Email:
{observation.email_text}

Return JSON format:
category:
priority:
action:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

text = response.choices[0].message.content

print(text)