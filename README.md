# Career Pivot Simulator (LLM-Enhanced Decision Engine)

AI-powered prototype for exploring realistic career transitions using occupational skill data and multi-agent LLM reasoning.

**Live App:**  
https://career-pivot-simulator.streamlit.app/

**Repository:**  
https://github.com/Thisisntevenmyfinale/career-pivot-simulator

Course: *Prototyping Products with Data and Artificial Intelligence*  
Program: Master in Business Analytics  
Instructor: Jose A. Rodriguez Serrano  

---

# Project Overview

The Career Pivot Simulator is a Streamlit prototype that helps users explore **realistic career transitions based on skill similarity and AI-supported reasoning**.

The first version of the prototype focused on:
- explainable career matching  
- skill gap analysis  
- stepping-stone routing  
- data-driven learning plans  

The extended version transforms this system into a **decision-support engine** by introducing a non-trivial LLM reasoning layer.

Instead of only answering *“what job fits?”*, the system now answers:

• What is realistically achievable?  
• Which strategy should I follow?  
• What would convince the market?  
• Where are the real risks?  
• What could change the recommendation?  

The result is not just a recommender system, but a **structured decision framework combining data models and LLM-based reasoning**.

---

# Key Features

## Data-Driven Career Matching

Occupations are represented as skill vectors derived from the O*NET database.

Cosine similarity is used to measure how close two careers are.

## Percentile Contextualization

Similarity scores are contextualized relative to all possible transitions.

## Skill Gap Analysis

**Transferable skills**  
skills that strongly overlap between roles.

**Missing skills**  
skills required in the target role but underdeveloped.

## Stepping-Stone Career Paths

Occupations are modeled as nodes in a similarity graph.

Shortest-path logic identifies realistic intermediate roles.

## Skill Investment Simulator

Users can simulate how improving specific skills changes their match score.

---

# LLM-Enhanced Features (Assignment 2)

## Multi-Strategy Generation

The system generates competing pivot strategies:
- Direct  
- Stepping-Stone  
- Skill-First  
- Portfolio-First  
- Hybrid  

Each includes structured attributes such as risk, speed, and feasibility.

## Multi-Agent Evaluation

Each strategy is evaluated by multiple personas:
- Hiring Manager  
- Recruiter  
- Risk Analyst  
- Portfolio Evaluator  
- Career Coach  

## Disagreement and Robustness Modeling

Outputs are processed in Python to compute:
- average scores  
- disagreement  
- robustness  
- confidence-adjusted rankings  

## Final Recommendation (Judge Layer)

A final synthesis produces:
- recommended strategy  
- risks  
- success conditions  
- decision sensitivity  

## AI Learning Plan

A structured learning plan is generated based on real skill gaps.

Fallback logic ensures robustness without API.

## AI Coach

Interactive assistant for quick validation and short-term guidance.

---

# System Architecture

O*NET Raw Data  
→ Offline preprocessing  
→ Skill Matrix + PCA + Clusters  
→ Streamlit App  

Data Layer:  
- similarity scoring  
- skill gaps  
- routing  

LLM Layer:  
- strategy generation  
- multi-agent evaluation  
- aggregation  
- decision synthesis  

---

# Why This Is Non-Trivial LLM Usage

The system uses:

• multiple coordinated LLM calls  
• role-based prompting  
• structured outputs processed in Python  
• aggregation and ranking logic  
• integration with deterministic models  

The LLM is part of a **decision pipeline**, not just a generator.

---

# Data Source

O*NET occupational database (U.S. Department of Labor)

---

# Technology Stack

Python  
Streamlit  
Pandas  
NumPy  
Scikit-learn  
NetworkX  

LLM API integration

---

# Repository Structure

career-pivot-simulator  
│  
├── app.py  
├── requirements.txt  
├── runtime.txt  
│  
├── src  
│   ├── ai_coach.py  
│   ├── decision_engine.py  
│   ├── map_pipeline.py  
│   ├── model_logic.py  
│   └── preprocessing.py  
│  
├── scripts  
├── data  
└── artifacts  

---

# Running Locally

git clone https://github.com/Thisisntevenmyfinale/career-pivot-simulator  
cd career-pivot-simulator  
pip install -r requirements.txt  
streamlit run app.py  

---

# Reflection

The system evolves from a recommender into a **decision-support system**.

Instead of predicting one answer, it:
- generates alternatives  
- evaluates trade-offs  
- models uncertainty  
- supports execution  

---

# Author

Jan Philipp Gnau  
Master in Business Analytics