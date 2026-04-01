---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---

# Email Triage OpenEnv Environment

## Overview
This project implements an OpenEnv-compatible environment simulating an email triage workflow.  
An AI agent processes incoming emails and predicts:

1. category
2. priority
3. action

The environment simulates real-world customer support automation.

---

## Tasks

### Easy Task
Predict email category:
- billing
- technical
- account
- general

### Medium Task
Predict:
- category
- priority (low, medium, high)

### Hard Task
Predict:
- category
- priority
- action (reply, archive, escalate)

---

## Reward Logic

Score range: 0–1

category correct → 0.33  
priority correct → 0.33  
action correct → 0.34  

Partial progress is rewarded.

---

## Environment API

reset() → returns email observation  

step(action) → returns reward score  

state() → returns current email state  

---

## Run locally

python inference.py

---

## Project Structure

environment.py → core environment logic  

tasks/ → task definitions  

dataset/ → email dataset  

inference.py → baseline agent  

Dockerfile → container setup  

openenv.yaml → environment specification  

---

## Motivation

Email triage is a real-world workflow used in:

customer support  

IT helpdesk  

HR communication  

finance operations  

Automating this process improves efficiency and reduces manual workload.
=======
---
title: Email Triage Openenv
emoji: 🔥
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: An AI agent processes incoming emails and predicts
---
