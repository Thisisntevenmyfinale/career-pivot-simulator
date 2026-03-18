"""
Pydantic schemas for Adversarial Review Board.
Ensures type safety + structured validation across the pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class StrategyArchetype(BaseModel):
    """One of five possible pivot strategy archetypes."""
    
    name: str = Field(..., description="Direct Pivot | Stepping-Stone | Skill-First | Portfolio-First | Hybrid")
    code: str = Field(..., description="DIRECT | STEPPING | SKILL_FIRST | PORTFOLIO | HYBRID")
    description: str
    estimated_days: int = Field(..., ge=14, le=365)
    risk_level: str = Field(..., description="low | medium | high")
    
    @validator('code')
    def validate_code(cls, v):
        allowed = {"DIRECT", "STEPPING", "SKILL_FIRST", "PORTFOLIO", "HYBRID"}
        if v not in allowed:
            raise ValueError(f"code must be one of {allowed}")
        return v
    
    @validator('risk_level')
    def validate_risk(cls, v):
        if v not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be low | medium | high")
        return v


class StrategyPhase(BaseModel):
    """One phase of a pivot strategy (0-30 days, 30-90 days, etc)."""
    
    phase: str = Field(..., description="e.g. '0-30 days', '30-90 days'")
    objective: str
    deliverables: List[str]
    key_actions: List[str]


class Strategy(BaseModel):
    """Complete pivot strategy with archetype, phases, and evidence."""
    
    archetype: StrategyArchetype
    current_role: str
    target_role: str
    summary: str = Field(..., max_length=500)
    phases: List[StrategyPhase] = Field(..., min_items=2, max_items=4)
    key_missing_skills: List[str] = Field(..., max_items=6)
    transferable_anchors: List[str] = Field(..., max_items=5)
    success_criteria: List[str] = Field(..., min_items=2, max_items=4)
    potential_risks: List[str] = Field(..., min_items=1, max_items=3)
    resources_needed: List[str] = Field(..., min_items=2, max_items=5)


class ReviewerScore(BaseModel):
    """One reviewer's evaluation of a single strategy."""
    
    reviewer_persona: str = Field(..., description="HiringManager | Recruiter | PortfolioEval | RiskAnalyst | CareerCoach")
    strategy_code: str  # e.g. "DIRECT", "HYBRID"
    overall_score: float = Field(..., ge=0.0, le=100.0)
    
    alignment_with_role: float = Field(..., ge=0.0, le=10.0, description="How well does this strategy prepare for target role?")
    market_feasibility: float = Field(..., ge=0.0, le=10.0, description="Is this realistic in job market?")
    time_efficiency: float = Field(..., ge=0.0, le=10.0, description="Time to target role (inverted: lower days = higher score)")
    risk_assessment: float = Field(..., ge=0.0, le=10.0, description="Lower risk = higher score")
    narrative_strength: float = Field(..., ge=0.0, le=10.0, description="Can candidate tell convincing story?")
    
    justification: str = Field(..., max_length=400)
    concerns: List[str] = Field(default_factory=list, max_items=3)
    
    @property
    def compute_overall(self) -> float:
        """Recompute overall from components if needed."""
        components = [
            self.alignment_with_role,
            self.market_feasibility,
            self.time_efficiency,
            self.risk_assessment,
            self.narrative_strength
        ]
        return sum(components) / len(components) * 10.0  # normalize to 100


class ReviewerEvaluation(BaseModel):
    """Full evaluation from one reviewer across all strategies."""
    
    reviewer_persona: str
    strategy_scores: List[ReviewerScore]
    overall_recommendation: str = Field(..., max_length=300)
    strongest_strategy: str  # e.g., "HYBRID"
    weakest_strategy: str


class ConsensusResult(BaseModel):
    """Aggregated consensus across all reviewers."""
    
    winner_strategy: str  # Code of best strategy
    winner_score: float
    runner_up_strategy: str
    runner_up_score: float
    
    consensus_strength: float = Field(..., ge=0.0, le=100.0, description="How much do reviewers agree?")
    major_disagreements: List[Dict[str, Any]] = Field(default_factory=list)
    
    strategy_rankings: List[tuple[str, float]]  # [(code, avg_score), ...]
    

class JudgeMemoRequest(BaseModel):
    """Input to final judge memo generation."""
    
    current_role: str
    target_role: str
    winner_strategy: str
    consensus_result: ConsensusResult
    gap_df_summary: Dict[str, Any]  # top missing, top transferable
    route_exists: bool


class JudgeMemo(BaseModel):
    """Final recommendation memo produced by judge LLM."""
    
    verdict: str  # "Highly Feasible" | "Feasible with Conditions" | "Challenging"
    recommended_strategy: str
    
    executive_summary: str = Field(..., max_length=600)
    key_success_factors: List[str]
    critical_risks: List[str]
    first_30_day_actions: List[str]
    
    interview_narrative: str = Field(..., max_length=500, description="How to pitch this pivot to hiring manager")
    success_timeline: str = Field(..., max_length=200, description="e.g., '6-9 months'")
    confidence_level: str = Field(..., description="High | Medium | Low")