# CTF LLM Project

## Overview

This project explores how large language models (LLMs) perform multi-step reasoning in CTF-style cybersecurity environments.

The main goal is to improve LLM-guided attack planning by giving the model a structured representation of the network state instead of relying only on short-term conversational context.

This project is inspired by a pivoting-style ethical hacking scenario in which an attacker must:

1. discover hosts on an external network  
2. identify vulnerable services  
3. exploit a foothold machine  
4. pivot through the compromised host  
5. reach a protected internal target  

---

## Research Focus

Initial testing showed that an LLM can often describe general penetration testing concepts, but struggles with:

- keeping track of discovered hosts
- maintaining awareness of the correct network
- selecting the most relevant next step
- avoiding hallucinated targets or subnets
- reasoning consistently across multiple actions

This project investigates whether **structured network-state tracking** improves LLM decision-making in cyber attack scenarios.

---

## Current Project Goal

Build an AI-assisted cybersecurity planning system that:

- stores the current network state
- tracks discovered hosts and services
- tracks blocked networks, pivot candidates, and compromised hosts
- provides structured context to an LLM
- evaluates whether that context improves attack-path reasoning

---

## Current Status

### Implemented
- project folder structure
- GitHub repository setup
- initial `NetworkState` model
- basic test/demo in `main.py`

### In Progress
- parser development for Nmap output
- automated network-state updates
- LLM planning integration

---

## Project Structure

```text
ctf-llm-project/
│
├── data/
├── logs/
├── src/
│   ├── experiments/
│   ├── models/
│   ├── parsers/
│   ├── prompts/
│   └── services/
├── tests/
├── main.py
├── README.md
└── requirements.txt