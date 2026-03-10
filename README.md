# Career Pivot Simulator

AI-powered prototype for exploring realistic career transitions using occupational skill data.

**Live App:**  
https://career-pivot-simulator.streamlit.app/

**Repository:**  
https://github.com/Thisisntevenmyfinale/career-pivot-simulator

Course: *Prototyping Products with Data and Artificial Intelligence*  
Program: Master in Business Analytics  
Instructor: Jose A. Rodriguez Serrano  

---

# Project Overview

The Career Pivot Simulator is a Streamlit prototype that helps users explore **realistic career transitions based on skill similarity**.

Instead of simply recommending jobs, the system explains:

• how similar two careers are  
• which skills transfer  
• which skills are missing  
• which intermediate careers could act as stepping stones  
• what a realistic learning plan could look like  

The goal is to prototype a **data-driven product feature** that could realistically be embedded into professional platforms such as LinkedIn.

The feature focuses on **decision support rather than black-box predictions**.

---

# Key Features

### Career Similarity Scoring
Occupations are represented as skill vectors derived from the O*NET database.

Cosine similarity is used to measure how close two careers are.

### Percentile Contextualization
Similarity scores are contextualized relative to all possible career transitions.

This makes the score interpretable for users.

### Skill Gap Analysis
The system identifies:

**Transferable skills**  
skills that both careers require strongly.

**Missing skills**  
skills that are important in the target role but weaker in the current role.

### Stepping-Stone Career Paths
Occupations are modeled as nodes in a similarity graph.

Shortest path algorithms identify realistic intermediate roles.

This helps users plan transitions step-by-step rather than jumping directly to a distant career.

### AI Learning Plan Generator
An optional AI component generates a structured **3-phase learning plan** based on missing skills.

If the API is unavailable, the system falls back to a rule-based plan.

---

# System Architecture

The prototype separates **offline preprocessing** from **runtime inference**.

```
O*NET Raw Data
      ↓
Offline preprocessing
      ↓
Skill Matrix
PCA Coordinates
Clusters
Artifacts
      ↓
Streamlit Application
      ↓
Career Matching
Skill Gap Analysis
Stepping-Stone Paths
AI Learning Plan
```

This design ensures that expensive computations are performed offline while the user interface remains responsive.

---

# Data Source

The prototype uses the **O*NET occupational database**, maintained by the U.S. Department of Labor.

O*NET provides structured information about:

• occupations  
• required skills  
• work activities  
• abilities  

Using this dataset allows the prototype to rely on **real labor market data** instead of synthetic examples.

---

# Technology Stack

Python  
Streamlit  
Pandas  
NumPy  
Scikit-learn  
NetworkX  

AI integration:

ChatGPT / LLM API (optional learning plan generation)

---

# Repository Structure

```
career-pivot-simulator
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── src
│   ├── ai_coach.py
│   ├── map_pipeline.py
│   ├── model_logic.py
│   └── preprocessing.py
│
├── scripts
│   ├── preprocess_onet.py
│   └── preprocess_dummy.py
│
├── data
│   └── skills_long.csv
│
└── artifacts
```

**src/**  
Core model logic and algorithms.

**scripts/**  
Offline preprocessing pipeline.

**artifacts/**  
Precomputed data artifacts loaded at runtime.

**app.py**  
Streamlit interface.

---

# Running the App Locally

Clone the repository:

```
git clone https://github.com/Thisisntevenmyfinale/career-pivot-simulator
cd career-pivot-simulator
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app.py
```

---

# Prototyping Goals

The prototype was designed to reflect the **three dimensions of AI prototyping discussed in the course**.

### User Experience
A structured interaction flow helps users explore and interpret career transitions.

### Data / Model Pipeline
Offline preprocessing is separated from runtime inference.

### Accuracy and Trust
Explainability is provided through skill gap analysis and contextualized similarity scores.

---

# AI Usage

AI was used in two ways:

1. **Development support**

ChatGPT assisted with debugging, code review, and iteration during development.

2. **Product feature**

The prototype integrates an AI coach that generates learning plans based on detected skill gaps.

This demonstrates how generative AI can be layered on top of structured data systems.

---

# Limitations

This prototype focuses on concept validation rather than production deployment.

Limitations include:

• simplified learning plan generation  
• limited occupation filtering  
• no real labor market demand data  
• limited personalization

Future work could integrate:

• job market demand signals  
• course recommendation systems  
• salary data  
• user skill profiles

---

# Author

Jan Philipp Gnau  
Master in Business Analytics
