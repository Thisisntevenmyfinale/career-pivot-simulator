"""
Pydantic schemas for the Career Pivot Decision Engine.

Design goals:
- strong structured validation
- backward compatibility with the existing app
- support richer multi-agent review outputs
- clean Pydantic v2 style
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ALLOWED_STRATEGY_CODES = {"DIRECT", "STEPPING", "SKILL_FIRST", "PORTFOLIO", "HYBRID"}
_ALLOWED_RISK = {"low", "medium", "high"}
_ALLOWED_CONFIDENCE = {"High", "Medium", "Low"}
_ALLOWED_VERDICTS = {"Highly Feasible", "Feasible with Conditions", "Challenging"}


class StrategyArchetype(BaseModel):
    """One career pivot strategy archetype."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Human-readable strategy name")
    code: str = Field(..., description="DIRECT | STEPPING | SKILL_FIRST | PORTFOLIO | HYBRID")
    description: str
    estimated_days: int = Field(..., ge=14, le=365)
    risk_level: str = Field(..., description="low | medium | high")

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in _ALLOWED_STRATEGY_CODES:
            raise ValueError(f"code must be one of {_ALLOWED_STRATEGY_CODES}")
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_risk(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in _ALLOWED_RISK:
            raise ValueError("risk_level must be low | medium | high")
        return v


class StrategyPhase(BaseModel):
    """One phase of a career pivot strategy."""

    model_config = ConfigDict(extra="ignore")

    phase: str = Field(..., description="e.g. '0-30 days', '30-90 days', '90+ days'")
    objective: str
    deliverables: List[str] = Field(default_factory=list)
    key_actions: List[str] = Field(default_factory=list)

    @field_validator("deliverables", "key_actions")
    @classmethod
    def clean_lists(cls, v: List[str]) -> List[str]:
        items = [str(x).strip() for x in (v or []) if str(x).strip()]
        return items[:6]


class Strategy(BaseModel):
    """Complete pivot strategy with structured signals."""

    model_config = ConfigDict(extra="ignore")

    archetype: StrategyArchetype
    current_role: str
    target_role: str
    summary: str = Field(..., max_length=700)
    phases: List[StrategyPhase] = Field(..., min_length=2, max_length=4)

    key_missing_skills: List[str] = Field(default_factory=list, max_length=6)
    transferable_anchors: List[str] = Field(default_factory=list, max_length=5)
    success_criteria: List[str] = Field(default_factory=list, min_length=2, max_length=5)
    potential_risks: List[str] = Field(default_factory=list, min_length=1, max_length=4)
    resources_needed: List[str] = Field(default_factory=list, min_length=2, max_length=6)

    # richer signals for aggregation + UI
    best_for_profile: str = Field(default="", max_length=240)
    evidence_strategy: str = Field(default="", max_length=320)
    key_tradeoff: str = Field(default="", max_length=220)
    confidence_rationale: str = Field(default="", max_length=320)

    speed_bias: float = Field(default=5.0, ge=0.0, le=10.0)
    risk_bias: float = Field(default=5.0, ge=0.0, le=10.0)
    evidence_burden: float = Field(default=5.0, ge=0.0, le=10.0)
    market_signal_strength: float = Field(default=5.0, ge=0.0, le=10.0)
    skill_gap_focus: float = Field(default=5.0, ge=0.0, le=10.0)

    @field_validator("key_missing_skills", "transferable_anchors", "success_criteria", "potential_risks", "resources_needed")
    @classmethod
    def clean_string_lists(cls, v: List[str]) -> List[str]:
        items = [str(x).strip() for x in (v or []) if str(x).strip()]
        return items


class ReviewerScore(BaseModel):
    """One reviewer's evaluation of one strategy."""

    model_config = ConfigDict(extra="ignore")

    reviewer_persona: str = Field(..., description="HiringManager | Recruiter | PortfolioEval | RiskAnalyst | CareerCoach")
    strategy_code: str
    overall_score: float = Field(..., ge=0.0, le=100.0)

    alignment_with_role: float = Field(..., ge=0.0, le=10.0)
    market_feasibility: float = Field(..., ge=0.0, le=10.0)
    time_efficiency: float = Field(..., ge=0.0, le=10.0)
    risk_assessment: float = Field(..., ge=0.0, le=10.0)
    narrative_strength: float = Field(..., ge=0.0, le=10.0)

    justification: str = Field(..., max_length=500)
    concerns: List[str] = Field(default_factory=list, max_length=4)

    # richer adversarial review fields
    best_strength: str = Field(default="", max_length=220)
    biggest_risk: str = Field(default="", max_length=220)
    killer_objection: str = Field(default="", max_length=220)
    success_condition: str = Field(default="", max_length=220)
    best_candidate_fit: str = Field(default="", max_length=220)

    @field_validator("strategy_code")
    @classmethod
    def validate_strategy_code(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in _ALLOWED_STRATEGY_CODES:
            raise ValueError(f"strategy_code must be one of {_ALLOWED_STRATEGY_CODES}")
        return v

    @property
    def compute_overall(self) -> float:
        components = [
            self.alignment_with_role,
            self.market_feasibility,
            self.time_efficiency,
            self.risk_assessment,
            self.narrative_strength,
        ]
        return sum(components) / len(components) * 10.0


class ReviewerEvaluation(BaseModel):
    """One reviewer across all strategies."""

    model_config = ConfigDict(extra="ignore")

    reviewer_persona: str
    strategy_scores: List[ReviewerScore]
    overall_recommendation: str = Field(..., max_length=400)
    strongest_strategy: str
    weakest_strategy: str
    reviewer_weight: float = Field(default=1.0, ge=0.5, le=2.0)

    @field_validator("strongest_strategy", "weakest_strategy")
    @classmethod
    def validate_codes(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in _ALLOWED_STRATEGY_CODES:
            raise ValueError(f"strategy code must be one of {_ALLOWED_STRATEGY_CODES}")
        return v


class ConsensusResult(BaseModel):
    """Aggregated consensus across all reviewer personas."""

    model_config = ConfigDict(extra="ignore")

    winner_strategy: str
    winner_score: float
    runner_up_strategy: str
    runner_up_score: float

    consensus_strength: float = Field(..., ge=0.0, le=100.0)
    robustness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    controversy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    fragile_winner: bool = False

    major_disagreements: List[Dict[str, Any]] = Field(default_factory=list)
    reviewer_alignment_summary: List[Dict[str, Any]] = Field(default_factory=list)
    strategy_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)

    strategy_rankings: List[Tuple[str, float]] = Field(default_factory=list)

    @field_validator("winner_strategy", "runner_up_strategy")
    @classmethod
    def validate_rank_codes(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in _ALLOWED_STRATEGY_CODES:
            raise ValueError(f"strategy code must be one of {_ALLOWED_STRATEGY_CODES}")
        return v


class JudgeMemoRequest(BaseModel):
    """Input structure for final judge memo generation."""

    model_config = ConfigDict(extra="ignore")

    current_role: str
    target_role: str
    winner_strategy: str
    consensus_result: ConsensusResult
    gap_df_summary: Dict[str, Any]
    route_exists: bool


class JudgeMemo(BaseModel):
    """Final recommendation memo produced by the judge layer."""

    model_config = ConfigDict(extra="ignore")

    verdict: str
    recommended_strategy: str

    executive_summary: str = Field(..., max_length=700)
    key_success_factors: List[str] = Field(default_factory=list)
    critical_risks: List[str] = Field(default_factory=list)
    first_30_day_actions: List[str] = Field(default_factory=list)

    interview_narrative: str = Field(..., max_length=550)
    success_timeline: str = Field(..., max_length=200)
    confidence_level: str

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        v = str(v).strip()
        if v not in _ALLOWED_VERDICTS:
            return "Feasible with Conditions"
        return v

    @field_validator("recommended_strategy")
    @classmethod
    def validate_recommended_strategy(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in _ALLOWED_STRATEGY_CODES:
            return "HYBRID"
        return v

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        v = str(v).strip().title()
        if v not in _ALLOWED_CONFIDENCE:
            return "Medium"
        return v