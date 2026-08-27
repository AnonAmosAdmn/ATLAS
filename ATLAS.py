#!/usr/bin/env python3
"""
ATLAS MULTI-SPACE LANGUAGE LABORATORY — v10
==========================================

A runnable Streamlit laboratory bridging conventional multilingual embeddings
with the richer ATLAS object/state architecture used in this project.

Supported experiment modes
--------------------------
1. Single word / utterance + each token
2. Word ↔ Word
3. Word → Many Languages
4. Utterance ↔ Utterance
5. Utterance → Many Languages
6. Language Map ↔ Language Map
7. Coordinate / Gap Discovery
8. Language Intelligence Benchmark
9. Active Native Language Maps

ATLAS object hierarchy
----------------------
TopLayerState
    ├── LanguageMapState
    ├── UtteranceState
    │      └── TokenState[]
    ├── typedSpaces
    ├── fibers
    ├── relations
    ├── hyperrelations
    ├── transformations
    ├── alignments
    ├── invariants
    ├── residuals
    ├── uncertainty
    ├── attention
    ├── comprehension
    ├── knowledgeDelta
    ├── provenance
    └── validationStatus

Important epistemic rule
------------------------
Most typed coordinates in this prototype are HEURISTIC / PROVISIONAL.
Embeddings and latent vectors are learned observation layers, not native ATLAS
typed coordinates. Display projections are never treated as native state.

Run
---
pip install -r requirements_atlas_multispace_lab.txt
streamlit run atlas_multispace_lab.py
"""

from __future__ import annotations

import io
import json
import math
import random
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Constants / schemas
# =============================================================================

APP_TITLE = "ATLAS Multi-Space Language Laboratory v10 — Active Native Language Maps"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LANGUAGES = {
    "English": "en",
    "Farsi / Persian": "fa",
    "Arabic": "ar",
    "Mandarin Chinese": "zh",
    "Latin": "la",
    "Sanskrit": "sa",
    "Urdu": "ur",
    "Hebrew": "he",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Turkish": "tr",
    "Navajo": "nv",
    "Cherokee": "chr",
    "Apache": "apa",
    "Sumerian": "sux",
    "Unknown / Other": "und",
}

LANGUAGE_FAMILIES = {
    "en": "Indo-European / Germanic",
    "fa": "Indo-European / Iranian",
    "ar": "Afro-Asiatic / Semitic",
    "zh": "Sino-Tibetan / Sinitic",
    "la": "Indo-European / Italic",
    "sa": "Indo-European / Indo-Aryan",
    "ur": "Indo-European / Indo-Aryan",
    "he": "Afro-Asiatic / Semitic",
    "ru": "Indo-European / Slavic",
    "ja": "Japonic",
    "ko": "Koreanic",
    "tr": "Turkic",
    "nv": "Na-Dené / Athabaskan",
    "chr": "Iroquoian",
    "apa": "Na-Dené / Athabaskan",
    "sux": "Language isolate",
    "und": "Unknown",
}

SPACE_ORDER = [
    "ORTH", "PHON", "LEX", "POS", "NOUN", "VERB", "ADJ", "ADV", "FUNC",
    "MORPH", "SYN", "ROLE", "REF", "SEM", "LOG", "SCOPE", "EPI", "PRAG",
    "INFO", "DISC", "SOC", "SOC_REL", "SELF", "CAUS", "TEMP", "AFFECT",
    "UNC", "ATT", "COH", "COMP", "KNOW", "INTEL", "WIS", "META",
]

SPACE_DESCRIPTIONS = {
    "ORTH": "Orthography / graphemics",
    "PHON": "Phonology / phonological proxy",
    "LEX": "Lexicon / lexical state",
    "POS": "Part-of-speech state",
    "NOUN": "Nominal-state manifold",
    "VERB": "Verbal-state manifold",
    "ADJ": "Adjectival-state manifold",
    "ADV": "Adverbial-state manifold",
    "FUNC": "Function-word manifold",
    "MORPH": "Morphology",
    "SYN": "Syntax",
    "ROLE": "Semantic roles / argument structure",
    "REF": "Reference / deixis / coreference",
    "SEM": "Semantics",
    "LOG": "Logic",
    "SCOPE": "Scope / operator structure",
    "EPI": "Epistemics",
    "PRAG": "Pragmatics",
    "INFO": "Information structure",
    "DISC": "Discourse",
    "SOC": "Sociolinguistics / register",
    "SOC_REL": "Social relationship geometry",
    "SELF": "Self-reference / self-model relation",
    "CAUS": "Causality",
    "TEMP": "Temporality",
    "AFFECT": "Affect",
    "UNC": "Uncertainty",
    "ATT": "Attention",
    "COH": "Coherence",
    "COMP": "Comprehension",
    "KNOW": "Knowledge",
    "INTEL": "Intelligence operations",
    "WIS": "Wisdom / judgment",
    "META": "Meta-representation",
}

SPACE_COLORS = {
    name: color for name, color in zip(
        SPACE_ORDER,
        (
            [
                "#94a3b8", "#a3a3a3", "#58cbed", "#67e8f9", "#60a5fa",
                "#818cf8", "#a78bfa", "#c084fc", "#d8b4fe", "#5eead4",
                "#82aaff", "#22d3ee", "#38bdf8", "#14b8a6", "#e98181",
                "#d493ff", "#7bd69d", "#f19a67", "#facc15", "#8da4c5",
                "#f472b6", "#ec4899", "#fb7185", "#f97316", "#efaa68",
                "#f0c778", "#e8cf6b", "#71dfff", "#9ed5ff", "#b2c6ff",
                "#86e4b2", "#bc9cf2", "#f2df94", "#c084fc", "#d946ef",
            ] * 2
        )[:len(SPACE_ORDER)]
    )
}


EXPERIMENT_MODES = [
    "Single word / utterance + each token",
    "Word ↔ Word",
    "Word → Many Languages",
    "Utterance ↔ Utterance",
    "Utterance → Many Languages",
    "Language Map ↔ Language Map",
    "Coordinate / Gap Discovery",
    "Language Intelligence Benchmark",
    "Active Native Language Maps",
]


# =============================================================================
# Data models
# =============================================================================

@dataclass
class Provenance:
    source: str = "ATLAS_BASE_TYPED_ENCODER_V10"
    method: str = "rule+distributional"
    model: str = "prototype"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "HYPOTHESIZED"

@dataclass
class ValidationState:
    admissibility: str = "OK"
    consistency: float = 0.70
    confidence: float = 0.60
    validationStatus: str = "PROVISIONAL"

@dataclass
class UncertaintyState:
    vector: dict[str, float]
    covariance: list[list[float]]
    entropy_proxy: float
    ambiguity_preserved: bool = True

@dataclass
class Relation:
    source: str
    target: str
    relation: str
    space: str
    weight: float
    status: str = "PROVISIONAL"
    evidence: str = ""

@dataclass
class Hyperrelation:
    id: str
    relation: str
    nodes: list[str]
    roles: dict[str, str]
    confidence: float
    status: str = "HYPOTHESIZED"

@dataclass
class Transformation:
    id: str
    source: str
    target: str
    operation: str
    affected_spaces: list[str]
    delta: dict[str, float]
    status: str = "DERIVED"

@dataclass
class Alignment:
    id: str
    members: list[str]
    languages: list[str]
    embedding_coherence: float
    per_space_coherence: dict[str, float]
    status: str = "CANDIDATE_ALIGNMENT"

@dataclass
class InvariantCandidate:
    id: str
    label: str
    support: list[str]
    shared_coordinates: dict[str, dict[str, float]]
    confidence: float
    status: str = "CANDIDATE_INVARIANT"

@dataclass
class Residual:
    member: str
    space: str
    coordinate: str
    value: float
    reference: float
    residual: float
    status: str = "UNEXPLAINED_OR_LANGUAGE_SPECIFIC"

@dataclass
class CoordinateCandidate:
    id: str
    proposed_name: str
    source_axis: str
    explained_variance: float
    max_existing_coordinate_correlation: float
    novelty_score: float
    positive_members: list[str]
    negative_members: list[str]
    supporting_members: int
    evidence_independence: float
    status: str = "CANDIDATE_NEW_COORDINATE"

@dataclass
class GapCandidate:
    id: str
    gap_type: str
    location: str
    magnitude: float
    persistence: float
    supporting_members: list[str]
    competing_explanations: list[str]
    recommended_test: str
    status: str = "PERSISTENT_GAP_CANDIDATE"

@dataclass
class TermProposal:
    id: str
    language: str
    concept_label: str
    reason: str
    supporting_invariant_spaces: list[str]
    residual_pressure: float
    proposed_placeholder: str
    status: str = "TERM_PROPOSAL_WARRANTED"

@dataclass
class DiscoveryReport:
    coordinate_candidates: list[CoordinateCandidate]
    gap_candidates: list[GapCandidate]
    term_proposals: list[TermProposal]
    excluded_coordinates: list[str]
    diagnostics: dict[str, Any]
    status: str = "PROVISIONAL_DISCOVERY"

@dataclass
class LexicalSense:
    id: str
    lemma: str
    pos: str
    definition: str
    examples: list[str]
    source: str
    status: str = "PROVISIONAL"

@dataclass
class LexicalEdge:
    source: str
    target: str
    relation: str
    source_sense: str = ""
    target_sense: str = ""
    weight: float = 0.50
    evidence: str = ""
    provenance: str = "ATLAS_LEXICAL_ENGINE_V10"
    status: str = "PROVISIONAL"

@dataclass
class LexicalNeighborhood:
    token_id: str
    surface: str
    lemma: str
    language: str
    senses: list[LexicalSense]
    edges: list[LexicalEdge]
    missing_relation_types: list[str]
    diagnostics: dict[str, Any]
    status: str = "PROVISIONAL"

@dataclass
class ReferenceEntity:
    id: str
    label: str
    person: str
    number: str
    discourse_role: str
    token_ids: list[str]
    confidence: float
    status: str = "PROVISIONAL"

@dataclass
class CoreferenceEdge:
    source: str
    target: str
    relation: str
    confidence: float
    evidence: str
    status: str = "PROVISIONAL"

@dataclass
class SemanticRoleBinding:
    predicate: str
    role: str
    filler: str
    confidence: float
    evidence: str
    status: str = "PROVISIONAL"

@dataclass
class MeaningFrame:
    id: str
    frame_type: str
    predicate: str
    predicate_token: str
    roles: list[SemanticRoleBinding]
    polarity: str
    modality: str
    affect_type: str
    affect_valence: float | None
    direction: str
    canonical_form: str
    confidence: float
    status: str = "PROVISIONAL"

@dataclass
class ComprehensionCertificate:
    predicate_resolved: bool
    subject_resolved: bool
    object_resolved: bool
    coreference_resolved: bool
    affect_target_resolved: bool
    speaker_resolved: bool
    addressee_resolved: bool
    scope_resolved: bool
    ambiguities: list[str]
    unresolved_items: list[str]
    confidence: float
    status: str = "PROVISIONAL"

@dataclass
class StateDelta:
    knowledge: float
    belief: float
    affect: float
    self_state: float
    social_relation: float
    pragmatic: float
    uncertainty: float
    status: str = "PROVISIONAL"

@dataclass
class CounterfactualNeighbor:
    text: str
    operation: str
    changed_dimensions: list[str]
    expected_invariants: list[str]
    expected_changes: dict[str, Any]
    status: str = "GENERATED_TEST"

@dataclass
class BenchmarkFinding:
    category: str
    label: str
    severity: str
    language: str
    utterance: str
    evidence: str
    recommendation: str
    status: str = "PROVISIONAL"

@dataclass
class LanguageIntelligenceProfile:
    language: str
    language_name: str
    utterance_count: int
    token_count: int
    lexical_coverage: float
    lexical_relation_density: float
    meaning_frame_coverage: float
    semantic_role_coverage: float
    reference_resolution: float
    coreference_resolution: float
    comprehension_confidence: float
    validation_confidence: float
    uncertainty: float
    typed_space_coverage: dict[str, float]
    frame_type_counts: dict[str, int]
    relation_type_counts: dict[str, int]
    lexical_relation_counts: dict[str, int]
    failure_count: int
    native_map_coverage: float = 0.0
    native_analyzer: str = ""
    status: str = "PROVISIONAL"

@dataclass
class IntelligenceBenchmarkReport:
    id: str
    title: str
    total_languages: int
    total_utterances: int
    total_tokens: int
    language_profiles: list[LanguageIntelligenceProfile]
    global_metrics: dict[str, float]
    typed_space_coverage: dict[str, float]
    frame_inventory: dict[str, int]
    structural_relation_inventory: dict[str, int]
    lexical_relation_inventory: dict[str, int]
    findings: list[BenchmarkFinding]
    cross_language_alignment: Alignment | None
    invariant_candidate: InvariantCandidate | None
    residual_count: int
    discovery: DiscoveryReport | None
    benchmark_dimensions: dict[str, float]
    status: str = "PROVISIONAL_INTELLIGENCE_REPORT"

@dataclass
class Token:
    id: str
    index_u: int
    surface: str
    lemma: str
    pos: str
    morph: str
    span: tuple[int, int]
    context_u: str
    language: str

@dataclass
class TokenState:
    token: Token
    typed_spaces: dict[str, dict[str, float]]
    relations: list[Relation]
    hyperrelations: list[Hyperrelation]
    uncertainty: UncertaintyState
    provenance: Provenance
    validation: ValidationState
    lexicalNeighborhood: LexicalNeighborhood | None = None

@dataclass
class UtteranceState:
    id: str
    form: str
    language: str
    tokens: list[TokenState]
    typed_spaces: dict[str, dict[str, float]]
    relations: list[Relation]
    hyperrelations: list[Hyperrelation]
    transformations: list[Transformation]
    candidateInterpretations: list[dict[str, Any]]
    uncertainty: UncertaintyState
    attention: dict[str, Any]
    comprehension: dict[str, Any]
    knowledgeDelta: dict[str, Any]
    provenance: Provenance
    validation: ValidationState
    references: list[ReferenceEntity] = field(default_factory=list)
    coreference: list[CoreferenceEdge] = field(default_factory=list)
    semanticRoles: list[SemanticRoleBinding] = field(default_factory=list)
    meaningFrames: list[MeaningFrame] = field(default_factory=list)
    comprehensionCertificate: ComprehensionCertificate | None = None
    stateDelta: StateDelta | None = None
    counterfactualNeighbors: list[CounterfactualNeighbor] = field(default_factory=list)
    languageMapAnalyzer: str = ""
    nativeMechanicsObserved: dict[str, Any] = field(default_factory=dict)

@dataclass
class LanguageMapState:
    id: str
    language: str
    language_name: str
    family: str
    dialectSet: list[str]
    orthography: dict[str, Any]
    phonology: dict[str, Any]
    lexicon: dict[str, Any]
    morphology: dict[str, Any]
    syntax: dict[str, Any]
    semantics: dict[str, Any]
    logic: dict[str, Any]
    epistemics: dict[str, Any]
    pragmatics: dict[str, Any]
    informationStructure: dict[str, Any]
    discourse: dict[str, Any]
    sociolinguistics: dict[str, Any]
    contextFibers: list[str]
    typed_spaces: dict[str, dict[str, float]]
    relations: list[Relation]
    hyperrelations: list[Hyperrelation]
    transformations: list[str]
    neighborhoods: dict[str, Any]
    density: dict[str, Any]
    topology: dict[str, Any]
    uncertainty: UncertaintyState
    provenance: Provenance
    validation: ValidationState
    nativeMechanics: dict[str, Any] = field(default_factory=dict)
    nativeOperators: list[dict[str, Any]] = field(default_factory=list)
    nativeConstraints: list[dict[str, Any]] = field(default_factory=list)
    nativeMetrics: dict[str, Any] = field(default_factory=dict)
    alignmentInterfaces: list[dict[str, Any]] = field(default_factory=list)
    analyzerStatus: dict[str, Any] = field(default_factory=dict)
    observationCoverage: dict[str, float] = field(default_factory=dict)

@dataclass
class TopLayerState:
    id: str
    languageMaps: list[LanguageMapState]
    typedSpaces: list[str]
    fibers: list[dict[str, Any]]
    metrics: dict[str, Any]
    relations: list[Relation]
    hyperrelations: list[Hyperrelation]
    transformations: list[Transformation]
    alignments: list[Alignment]
    invariants: list[InvariantCandidate]
    residuals: list[Residual]
    uncertainty: dict[str, Any]
    attention: dict[str, Any]
    comprehension: dict[str, Any]
    knowledgeDelta: dict[str, Any]
    provenance: Provenance
    validationStatus: ValidationState
    coordinateCandidates: list[CoordinateCandidate] = field(default_factory=list)
    gapCandidates: list[GapCandidate] = field(default_factory=list)
    termProposals: list[TermProposal] = field(default_factory=list)


# =============================================================================
# Core utilities
# =============================================================================

def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))

def mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else float(default)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

def has(text: str, cues: Iterable[str]) -> bool:
    """
    Boundary-aware cue matcher.

    Important:
        "no" must NOT match "know".
        "he" must NOT match "the".

    A cue ending in "*" is an explicit stem/prefix cue.
    For Han/Kana/Hangul/Arabic-script text we allow contained multi-character
    matches because segmentation is not always whitespace-delimited.
    """
    t = text.casefold().strip()
    if not t:
        return False

    script = script_profile(t)
    compact_script = (
        script["han"] > 0.4 or
        script["kana"] > 0.4 or
        script["hangul"] > 0.4 or
        script["arabic"] > 0.4
    )

    for cue in cues:
        c = cue.casefold().strip()
        if not c:
            continue

        if c.endswith("*"):
            if t.startswith(c[:-1]):
                return True
            continue

        if t == c:
            return True

        # Phrases are allowed inside longer utterance fragments.
        if " " in c and c in t:
            return True

        # For scripts where whitespace tokenization is unreliable, allow
        # multi-character contained cues, but never one-character accidental
        # matches.
        if compact_script and len(c) >= 2 and c in t:
            return True

    return False

def script_profile(text: str) -> dict[str, float]:
    groups = {
        "latin": 0, "arabic": 0, "cyrillic": 0, "han": 0,
        "hangul": 0, "kana": 0, "devanagari": 0, "other": 0
    }
    total = 0
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        total += 1
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            groups["latin"] += 1
        elif "ARABIC" in name:
            groups["arabic"] += 1
        elif "CYRILLIC" in name:
            groups["cyrillic"] += 1
        elif "CJK" in name or "IDEOGRAPH" in name:
            groups["han"] += 1
        elif "HANGUL" in name:
            groups["hangul"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:
            groups["kana"] += 1
        elif "DEVANAGARI" in name:
            groups["devanagari"] += 1
        else:
            groups["other"] += 1
    if total == 0:
        return {k: 0.0 for k in groups}
    return {k: v / total for k, v in groups.items()}

def punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    p = sum(unicodedata.category(ch).startswith("P") for ch in text)
    return p / max(1, len(text))

def simple_entropy(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = np.clip(vals, 1e-8, None)
    vals = vals / vals.sum() if vals.sum() else np.ones_like(vals) / len(vals)
    return float(-(vals * np.log(vals)).sum())


# =============================================================================
# Cue registries
# =============================================================================

NEG = (
    "not","no","never","n't","نه","نیست","نمی","لا","ليس","لم","لن",
    "不","没","沒有","не","нет","ない","ません","아니","않","न","नहीं",
    "לא","אין","değil","yok"
)
MODAL = (
    "may","might","perhaps","possibly","possible","could","شاید","ممکن",
    "ربما","قد","ممكن","可能","也许","也許","возможно","может","かもしれ",
    "たぶん","아마","수도","शायद","संभव","אולי","אפשר","belki","mümkün"
)
CERTAIN = (
    "certain","certainly","definitely","know","knows","known","مطمئن","قطعاً",
    "می‌دان","میدان","بالتأكيد","أعلم","نعلم","确定","確定","知道","точно",
    "знаю","известно","確実","知って","확실","알고","निश्चित","जानता",
    "בטוח","יודע","kesin","biliyorum"
)
BELIEF = (
    "believe","think","suspect","assume","باور","فکر","گمان","أعتقد","أظن",
    "افترض","认为","認為","相信","猜","думаю","верю","предполага","思う",
    "信じ","推測","생각","믿","추정","मानता","सोच","अनुमान","חושב",
    "מאמין","מניח","inan","düşün","san"
)
EVIDENCE = (
    "evidence","observe","observed","measurement","measured","data","proof",
    "مدرک","شواهد","داده","اندازه","دليل","أدلة","بيانات","قياس","证据",
    "證據","数据","數據","测量","測量","доказ","данные","измер","証拠",
    "データ","測定","증거","데이터","측정","प्रमाण","डेटा","מ","ראיה",
    "נתונים","מדידה","kanıt","veri","ölç"
)
CAUSAL = (
    "cause","caused","because","therefore","result","effect","leads to","علت",
    "باعث","زیرا","بنابراین","نتیجه","سبب","لأن","لذلك","نتيجة","因为",
    "因為","导致","導致","所以","结果","結果","потому","причин","поэтому",
    "результ","ため","原因","結果","때문","원인","결과","कारण","क्योंकि",
    "इसलिए","परिणाम","בגלל","סיבה","לכן","תוצאה","çünkü","neden","sonuç"
)
TEMPORAL = (
    "before","after","when","while","during","already","yet","future","past",
    "قبل","بعد","زمان","هنگام","هنوز","آینده","گذشته","عندما","أثناء",
    "المستقبل","الماضي","之前","之后","之後","当","當","未来","未來",
    "过去","過去","до","после","когда","будущ","прошл","前","後","時",
    "미래","과거","전","후","पहले","बाद","भविष्य","अतीत","לפני","אחרי",
    "עתיד","עבר","önce","sonra","gelecek","geçmiş"
)
POS_AFFECT = (
    "love","care","joy","happy","trust","kind","عشق","دوست","شاد","اعتماد",
    "حب","سعيد","ثقة","爱","愛","快乐","快樂","信任","люб","счаст","довер",
    "嬉","信頼","사랑","행복","신뢰","प्रेम","खुश","विश्वास","אהב","שמח",
    "אמון","sev","mutlu","güven"
)
NEG_AFFECT = (
    "hate","fear","angry","sad","hurt","disgust","نفرت","ترس","عصبانی",
    "غم","كره","خوف","غاضب","حزين","恨","怕","愤怒","憤怒","悲伤","悲傷",
    "ненав","страх","зл","груст","嫌","怖","怒","悲","싫","두려","화",
    "슬프","घृणा","डर","क्रोध","दुख","שנא","פחד","כעס","עצב","nefret",
    "kork","kızgın","üzgün"
)
QUESTION = (
    "what","why","how","who","where","when","?","چه","چرا","چگونه","کی","کجا",
    "؟","ماذا","لماذا","كيف","من","أين","متى","什么","为什么","為什麼",
    "怎么","怎麼","谁","誰","哪里","哪裡","что","почему","как","кто","где",
    "何","なぜ","どう","誰","どこ","무엇","왜","어떻게","누구","어디",
    "क्या","क्यों","कैसे","कौन","कहाँ","מה","למה","איך","מי","איפה",
    "ne","neden","nasıl","kim","nerede"
)
DISCOURSE = (
    "but","although","however","therefore","yet","because","so","اما","اگرچه",
    "بنابراین","زیرا","لكن","ومع ذلك","لذلك","لأن","但是","然而","所以",
    "因为","因為","но","однако","поэтому","потому","しかし","でも","だから",
    "하지만","그러나","그래서","लेकिन","हालाँकि","इसलिए","क्योंकि","אבל",
    "למרות","לכן","כי","ama","ancak","bu yüzden","çünkü"
)
META = (
    "meaning","word","sentence","utterance","language","interpret","model",
    "representation","state","semantic","logic","epistemic","atlas","معنی","واژه",
    "جمله","زبان","مدل","حالت","معنى","كلمة","جملة","لغة","نموذج","意义",
    "意義","词","詞","句子","语言","語言","模型","значение","слово","язык",
    "модель","意味","単語","文","言語","モデル","의미","단어","문장","언어",
    "मॉडल","अर्थ","शब्द","वाक्य","भाषा","משמעות","מילה","משפט","שפה",
    "anlam","kelime","cümle","dil","model"
)


# =============================================================================
# Tokenization / lexical class
# =============================================================================

TOKEN_RE = re.compile(r"""[\w]+(?:['’\-][\w]+)*|[^\w\s]""", re.UNICODE)

def tokenize(text: str, lang: str, utt_id: str) -> list[Token]:
    tokens = []
    for m in TOKEN_RE.finditer(text):
        surface = m.group(0)
        if not surface.strip():
            continue
        lemma = surface.casefold()
        pos = guess_pos(surface)
        morph = guess_morph(surface, pos)
        tokens.append(
            Token(
                id=f"t_{len(tokens)}",
                index_u=len(tokens),
                surface=surface,
                lemma=lemma,
                pos=pos,
                morph=morph,
                span=(m.start(), m.end()),
                context_u=utt_id,
                language=lang,
            )
        )
    return tokens

def guess_pos(surface: str) -> str:
    w = surface.casefold()

    if len(surface) == 1 and unicodedata.category(surface).startswith("P"):
        return "PUNCT"

    # English structure-first lexical classes before loose cue heuristics.
    if w in ENGLISH_PRONOUNS:
        return "PRON"
    if w in ENGLISH_DETERMINERS:
        return "DET"
    if w in ENGLISH_AUX:
        return "AUX"
    if w in ENGLISH_CONJ:
        return "SCONJ"
    if w in ENGLISH_COMMON_VERBS:
        return "VERB"
    if w in ENGLISH_COMMON_NOUNS:
        return "NOUN"
    if w in ENGLISH_COMMON_ADJ:
        return "ADJ"

    # Curated lexical registry can provide a strong POS prior.
    if w in CURATED_LEXICON_V7:
        return CURATED_LEXICON_V7[w].get("pos", "X")

    if has(w, NEG):
        return "PART"
    if has(w, MODAL):
        return "AUX"
    if has(w, DISCOURSE):
        return "SCONJ"
    if re.search(r"(ing|ed|ize|ise|fy)$", w) or has(w, BELIEF + CAUSAL):
        return "VERB"
    if re.search(r"(ly)$", w):
        return "ADV"
    if re.search(r"(ous|ful|ive|al|ic|able|ible|less|ary|ory)$", w):
        return "ADJ"
    if re.search(r"(tion|sion|ment|ness|ity|ism|ship|ance|ence|er|or)$", w):
        return "NOUN"
    if len(w) <= 3:
        return "FUNC"
    return "X"

def guess_morph(surface: str, pos: str) -> str:
    w = surface.casefold()
    feats = []
    if pos == "VERB":
        if w.endswith("ing"):
            feats += ["VerbForm=Part", "Aspect=Prog"]
        elif w.endswith("ed"):
            feats += ["Tense=Past"]
        else:
            feats += ["VerbForm=Fin"]
    if w.endswith("s") and len(w) > 3:
        feats += ["Number/Agreement=Possible"]
    if "-" in w:
        feats += ["Compound=Possible"]
    return "|".join(feats) if feats else "_"


# =============================================================================
# Typed-state encoder
# =============================================================================

def token_space_state(tok: Token, text: str, lang: str) -> dict[str, dict[str, float]]:
    w = tok.lemma
    sp = script_profile(tok.surface)
    punct = 1.0 if tok.pos == "PUNCT" else 0.0

    neg = has(w, NEG)
    modal = has(w, MODAL)
    certain = has(w, CERTAIN)
    belief = has(w, BELIEF)
    evidence = has(w, EVIDENCE)
    causal = has(w, CAUSAL)
    temporal = has(w, TEMPORAL)
    q = has(w, QUESTION)
    d = has(w, DISCOURSE)
    meta = has(w, META)
    pos_aff = has(w, POS_AFFECT)
    neg_aff = has(w, NEG_AFFECT)

    states: dict[str, dict[str, float]] = {}

    states["ORTH"] = {
        "graphemic_complexity": clamp(len(tok.surface) / 14),
        "punctuation": punct,
        "script_latin": sp["latin"],
        "script_arabic": sp["arabic"],
        "script_cyrillic": sp["cyrillic"],
        "script_han": sp["han"],
        "script_kana": sp["kana"],
        "script_hangul": sp["hangul"],
        "script_devanagari": sp["devanagari"],
    }

    states["PHON"] = {
        "phonological_observability": 0.20,
        "prosody_dependency": clamp(0.22 + (0.24 if q or d else 0.0)),
        "stress_tone_unknown": 0.85,
    }

    lexical_specificity = 0.05 if punct else clamp(0.22 + min(len(w), 16) / 24)
    states["LEX"] = {
        "specificity": lexical_specificity,
        "content_likelihood": 0.10 if punct else (0.35 if tok.pos in {"FUNC","PART","AUX","SCONJ"} else 0.78),
        "sense_ambiguity": clamp(0.28 + (0.12 if len(w) <= 5 else 0.0)),
        "frequency_unknown": 0.75,
    }

    states["POS"] = {
        "noun": 1.0 if tok.pos == "NOUN" else 0.0,
        "verb": 1.0 if tok.pos == "VERB" else 0.0,
        "adj": 1.0 if tok.pos == "ADJ" else 0.0,
        "adv": 1.0 if tok.pos == "ADV" else 0.0,
        "pron": 1.0 if tok.pos == "PRON" else 0.0,
        "det": 1.0 if tok.pos == "DET" else 0.0,
        "function": 1.0 if tok.pos in {"FUNC","PART","AUX","SCONJ","PRON","DET"} else 0.0,
        "unknown": 1.0 if tok.pos == "X" else 0.0,
    }

    states["NOUN"] = {
        "nominality": 1.0 if tok.pos == "NOUN" else 0.15,
        "referentiality_proxy": 0.70 if tok.pos == "NOUN" else 0.25,
        "entity_likelihood": 0.68 if tok.pos == "NOUN" else 0.20,
    }
    states["VERB"] = {
        "verbality": 1.0 if tok.pos == "VERB" else (0.72 if tok.pos == "AUX" else 0.12),
        "event_likelihood": clamp(0.65 if tok.pos == "VERB" else 0.15),
        "predicate_likelihood": clamp(0.80 if tok.pos in {"VERB","AUX"} else 0.18),
    }
    states["ADJ"] = {
        "adjectivality": 1.0 if tok.pos == "ADJ" else 0.10,
        "property_likelihood": 0.72 if tok.pos == "ADJ" else 0.18,
        "gradable_proxy": 0.50 if tok.pos == "ADJ" else 0.12,
    }
    states["ADV"] = {
        "adverbiality": 1.0 if tok.pos == "ADV" else 0.10,
        "modifier_likelihood": 0.72 if tok.pos == "ADV" else 0.16,
        "scope_modifier": 0.62 if tok.pos == "ADV" else 0.12,
    }
    states["FUNC"] = {
        "functional_weight": 0.90 if tok.pos in {"FUNC","PART","AUX","SCONJ"} else 0.10,
        "grammaticalization_proxy": 0.75 if tok.pos in {"PART","AUX","SCONJ"} else 0.18,
        "content_suppression": 0.72 if tok.pos in {"FUNC","PART","AUX","SCONJ"} else 0.12,
    }

    states["MORPH"] = {
        "complexity": clamp(0.15 + (0.20 if "-" in w else 0) + (0.18 if len(w) > 9 else 0)),
        "inflection_likelihood": clamp(0.15 + (0.42 if re.search(r"(s|ed|ing|en|er|est)$", w) else 0)),
        "derivation_likelihood": clamp(0.12 + (0.40 if re.search(r"(ness|ment|tion|ity|ism|able|less|ful)$", w) else 0)),
        "agreement_unknown": 0.72,
    }

    states["SYN"] = {
        "predicate_likelihood": states["VERB"]["predicate_likelihood"],
        "operator_likelihood": clamp(0.12 + (0.72 if neg or modal else 0)),
        "connector_likelihood": clamp(0.10 + (0.78 if d else 0)),
        "dependency_uncertainty": 0.45,
    }

    reflexive = w in {
        "myself","yourself","yourselves","himself","herself","itself",
        "ourselves","themselves"
    }
    first_person = w in {"i","me","my","mine","myself","we","us","our","ours","ourselves"}
    second_person = w in {"you","your","yours","yourself","yourselves"}

    states["ROLE"] = {
        "predicate_centrality": 0.92 if tok.pos == "VERB" else (0.45 if tok.pos in {"AUX","ADJ"} else 0.10),
        "experiencer_likelihood": 0.90 if first_person and tok.pos == "PRON" else 0.12,
        "target_likelihood": 0.88 if reflexive or second_person else (0.32 if tok.pos in {"NOUN","PRON"} else 0.10),
        "argument_salience": 0.82 if tok.pos in {"PRON","NOUN"} else 0.18,
    }

    states["REF"] = {
        "referentiality": 0.95 if tok.pos in {"PRON","NOUN","DET"} else 0.10,
        "first_person": 1.0 if first_person else 0.0,
        "second_person": 1.0 if second_person else 0.0,
        "reflexive": 1.0 if reflexive else 0.0,
        "speaker_role": 0.95 if first_person and tok.pos == "PRON" else 0.0,
        "addressee_role": 0.95 if second_person else 0.0,
        "coreference_pressure": 0.95 if reflexive else 0.08,
    }

    states["SELF"] = {
        "self_reference": 0.95 if first_person else 0.05,
        "self_target": 0.95 if reflexive else 0.05,
        "self_other_boundary": 0.20 if reflexive else (0.85 if second_person else 0.50),
        "identity_salience": 0.82 if first_person or reflexive else 0.12,
        "self_model_update_pressure": 0.70 if reflexive and (pos_aff or neg_aff) else 0.10,
    }

    states["SOC_REL"] = {
        "interpersonal_target": 0.92 if second_person else 0.10,
        "self_relational": 0.92 if reflexive else 0.08,
        "affinity": 0.88 if pos_aff else (0.12 if neg_aff else 0.50),
        "hostility": 0.88 if neg_aff else 0.08,
        "social_distance": 0.25 if reflexive else (0.55 if second_person else 0.50),
    }

    states["SEM"] = {
        "content_strength": 0.10 if punct else 0.68,
        "eventiveness": states["VERB"]["event_likelihood"],
        "abstractness": clamp(0.28 + (0.38 if meta or belief or certain else 0)),
        "relational_salience": clamp(0.20 + (0.42 if causal or belief else 0)),
    }

    states["LOG"] = {
        "negation": 1.0 if neg else 0.0,
        "truth_conditional_pressure": clamp(0.18 + (0.38 if certain or evidence else 0)),
        "operator_strength": clamp((0.88 if neg else 0) + (0.60 if modal else 0)),
        "entailment_unknown": 0.72,
    }

    states["SCOPE"] = {
        "scope_operator": 0.95 if neg else (0.82 if modal else 0.08),
        "scope_ambiguity": clamp(0.16 + (0.34 if neg or modal else 0)),
        "attachment_pressure": clamp(0.14 + (0.32 if d else 0)),
    }

    commitment = 0.50
    if certain:
        commitment = 0.90
    elif belief:
        commitment = 0.58
    elif modal:
        commitment = 0.32
    elif evidence:
        commitment = 0.72

    states["EPI"] = {
        "certainty": clamp(commitment),
        "evidence": 0.85 if evidence else 0.24,
        "commitment": commitment,
        "factivity_pressure": 0.82 if has(w, ("know","دان","知道","зна","ידע","知","bil")) else 0.22,
    }

    states["PRAG"] = {
        "force_assertive": clamp(0.65 - (0.30 if q else 0) - (0.15 if modal else 0)),
        "force_question": 0.95 if q else 0.05,
        "politeness_unknown": 0.55,
        "target_salience": 0.30,
    }

    states["INFO"] = {
        "topic_likelihood": clamp(0.32 + (0.10 if tok.index_u < 2 else 0)),
        "focus_likelihood": clamp(0.35 + (0.24 if evidence or neg or causal else 0)),
        "given_new_unknown": 0.62,
        "contrastive_focus": 0.82 if d else 0.08,
    }

    states["DISC"] = {
        "topic_link": 0.35,
        "cohesion": clamp(0.45 + (0.20 if d else 0)),
        "reference_unknown": 0.55,
        "connector_strength": 0.90 if d else 0.08,
    }

    states["SOC"] = {
        "register_unknown": 0.65,
        "formality_proxy": 0.50,
        "vernacularity_proxy": 0.35,
        "social_indexicality_unknown": 0.70,
    }

    states["CAUS"] = {
        "cause": 0.88 if causal else 0.10,
        "effect": 0.62 if has(w, ("result","effect","نتیجه","结果","результ","sonuç")) else 0.10,
        "direction": 0.68 if causal else 0.10,
    }

    states["TEMP"] = {
        "time_salience": 0.88 if temporal else 0.10,
        "aspect_unknown": 0.58,
        "sequence": 0.70 if temporal else 0.12,
    }

    valence = 0.85 if pos_aff else (0.15 if neg_aff else 0.50)
    states["AFFECT"] = {
        "valence": valence,
        "arousal": 0.70 if neg_aff else (0.55 if pos_aff else 0.20),
        "stance": 0.78 if pos_aff or neg_aff else 0.16,
        "positive_affect": 0.92 if pos_aff else 0.08,
        "negative_affect": 0.92 if neg_aff else 0.08,
        "approach": 0.88 if pos_aff else (0.18 if neg_aff else 0.50),
        "avoidance": 0.88 if neg_aff else 0.12,
        "targeted_affect": 0.88 if (pos_aff or neg_aff) else 0.10,
    }

    uncertainty = clamp(0.18 + (0.48 if modal else 0) + (0.18 if belief else 0) + (0.20 if q else 0) - (0.14 if certain else 0))
    states["UNC"] = {
        "lexical": states["LEX"]["sense_ambiguity"],
        "semantic": clamp(0.22 + (0.20 if modal else 0)),
        "context": 0.42,
        "scope": states["SCOPE"]["scope_ambiguity"],
        "aggregate": uncertainty,
    }

    states["ATT"] = {
        "salience": clamp(mean([
            states["LOG"]["operator_strength"],
            states["EPI"]["evidence"],
            states["CAUS"]["cause"],
            states["TEMP"]["time_salience"],
            states["AFFECT"]["stance"],
            0.90 if meta else 0.08,
        ]) + 0.10),
        "priority": clamp(0.20 + states["UNC"]["aggregate"] * 0.28 + states["EPI"]["evidence"] * 0.32),
        "novelty_unknown": 0.58,
    }

    states["COH"] = {
        "local": clamp(0.58 - states["UNC"]["aggregate"] * 0.12),
        "global": 0.48,
        "logical": clamp(0.58 - states["SCOPE"]["scope_ambiguity"] * 0.12),
    }

    states["COMP"] = {
        "syntax": clamp(1 - states["SYN"]["dependency_uncertainty"] * 0.50),
        "semantics": clamp(states["SEM"]["content_strength"] * 0.60 + 0.28),
        "scope": clamp(1 - states["SCOPE"]["scope_ambiguity"] * 0.55),
        "integration": clamp(mean([states["COH"]["local"], 1 - uncertainty, states["ATT"]["salience"]])),
    }

    states["KNOW"] = {
        "factivity": states["EPI"]["factivity_pressure"],
        "grounding": clamp(states["EPI"]["evidence"] * 0.82),
        "updateability": clamp(0.32 + states["UNC"]["aggregate"] * 0.28 + states["EPI"]["evidence"] * 0.30),
    }

    states["INTEL"] = {
        "pattern": clamp(0.22 + states["SEM"]["relational_salience"] * 0.28),
        "transfer": 0.35,
        "inference": clamp(0.20 + states["LOG"]["truth_conditional_pressure"] * 0.30 + states["CAUS"]["cause"] * 0.25),
    }

    states["WIS"] = {
        "calibration": clamp(0.28 + states["EPI"]["evidence"] * 0.25 + (1 - uncertainty) * 0.25),
        "consequence": clamp(0.12 + states["CAUS"]["cause"] * 0.30 + states["AFFECT"]["stance"] * 0.20),
        "restraint": clamp(0.28 + states["UNC"]["aggregate"] * 0.25),
        "context_sensitivity": clamp(0.30 + states["UNC"]["context"] * 0.32),
    }

    states["META"] = {
        "meta_salience": 0.90 if meta else 0.08,
        "representation_reference": 0.82 if meta else 0.10,
        "self_model_pressure": 0.70 if has(w, ("atlas","model","state","representation")) else 0.10,
    }

    return states



# =============================================================================
# Lexical relation engine
# =============================================================================

LEXICAL_RELATION_REGISTRY = {
    "SYNONYM": "same/near lexical sense",
    "ANTONYM": "lexical opposition",
    "POLYSEMOUS_SENSE": "related senses of one lexical form",
    "HOMONYM_CANDIDATE": "same form with potentially unrelated senses",
    "HOMOGRAPH_CANDIDATE": "same spelling; potentially different lexical item",
    "HOMOPHONE": "same pronunciation, different lexical item",
    "HYPERNYM": "is-a superclass",
    "HYPONYM": "is-a subtype",
    "COORDINATE_TERM": "shares a hypernym / taxonomic sibling",
    "MERONYM_PART": "part of",
    "MERONYM_MEMBER": "member of",
    "MERONYM_SUBSTANCE": "substance/component of",
    "HOLONYM_PART": "has part",
    "HOLONYM_MEMBER": "has member",
    "HOLONYM_SUBSTANCE": "has substance/component",
    "DERIVATIONALLY_RELATED": "same derivational family",
    "PERTAINYM": "pertains to",
    "ATTRIBUTE": "attribute relation",
    "ENTAILMENT": "lexical entailment",
    "CAUSE": "lexical causation",
    "TROPONYM": "manner-of verb relation",
    "MORPHOLOGICAL_RELATIVE": "shared root/stem candidate",
    "SEMANTIC_NEIGHBOR": "distributional neighbor; not asserted synonym",
    "COLLOCATION": "frequent lexical co-occurrence",
    "IDIOMATIC_RELATION": "participates in idiomatic/multiword expression",
    "TRANSLATION_EQUIVALENT": "cross-language lexical alignment",
    "COGNATE_CANDIDATE": "historical/formal cognacy candidate",
    "FALSE_FRIEND_CANDIDATE": "formally similar cross-language non-equivalence candidate",
}


# Minimal built-in lexical seed registry used only when WordNet/OMW is unavailable.
# These entries are intentionally small and explicitly tagged as CURATED_FALLBACK.
# They prevent the lab from degrading to "RESOURCE_UNAVAILABLE" for core benchmark
# vocabulary while preserving provenance.
CURATED_LEXICON_V7 = {
    "believe": {
        "pos": "VERB",
        "senses": [
            ("believe.v.01", "accept as true or probably true"),
            ("believe.v.02", "hold an opinion or conviction")
        ],
        "SYNONYM": ["think", "suppose", "accept", "trust"],
        "ANTONYM": ["disbelieve", "doubt"],
        "HYPERNYM": ["cognitive attitude"],
        "DERIVATIONALLY_RELATED": ["belief", "believer"],
        "COORDINATE_TERM": ["know", "suspect", "assume", "doubt"],
    },
    "model": {
        "pos": "NOUN",
        "senses": [
            ("model.n.01", "a representation of a system, object, or process"),
            ("model.n.02", "an example or pattern used for imitation")
        ],
        "SYNONYM": ["representation", "simulation", "pattern"],
        "HYPERNYM": ["representation"],
        "HYPONYM": ["mathematical model", "statistical model", "language model"],
        "DERIVATIONALLY_RELATED": ["modeling", "modelled"],
    },
    "understand": {
        "pos": "VERB",
        "senses": [
            ("understand.v.01", "grasp the meaning, significance, or structure of something")
        ],
        "SYNONYM": ["comprehend", "grasp", "apprehend"],
        "ANTONYM": ["misunderstand", "misinterpret"],
        "HYPERNYM": ["cognition"],
        "DERIVATIONALLY_RELATED": ["understanding"],
        "COORDINATE_TERM": ["know", "interpret", "infer", "recognize"],
    },
    "evidence": {
        "pos": "NOUN",
        "senses": [
            ("evidence.n.01", "information indicating whether a proposition is supported")
        ],
        "SYNONYM": ["support", "indication", "proof"],
        "ANTONYM": ["refutation"],
        "HYPERNYM": ["information"],
        "HYPONYM": ["testimony", "measurement", "observation", "data"],
        "COORDINATE_TERM": ["reason", "argument", "premise"],
    },
    "know": {
        "pos": "VERB",
        "senses": [
            ("know.v.01", "possess knowledge or justified awareness of something"),
            ("know.v.02", "be familiar with a person, fact, or domain")
        ],
        "SYNONYM": ["understand", "recognize", "be aware"],
        "ANTONYM": ["ignore", "misunderstand"],
        "HYPERNYM": ["cognition"],
        "DERIVATIONALLY_RELATED": ["knowledge", "knowing"],
        "COORDINATE_TERM": ["believe", "suspect", "doubt", "learn"],
    },
    "conclusion": {
        "pos": "NOUN",
        "senses": [
            ("conclusion.n.01", "a proposition reached by reasoning from premises"),
            ("conclusion.n.02", "the end or final part of something")
        ],
        "SYNONYM": ["inference", "judgment", "finding", "ending"],
        "ANTONYM": ["premise", "beginning"],
        "HYPERNYM": ["proposition"],
        "COORDINATE_TERM": ["premise", "claim", "hypothesis", "result"],
        "DERIVATIONALLY_RELATED": ["conclude"],
    },
    "true": {
        "pos": "ADJ",
        "senses": [
            ("true.a.01", "in accordance with fact or reality"),
            ("true.a.02", "accurate or correct")
        ],
        "SYNONYM": ["correct", "accurate", "valid"],
        "ANTONYM": ["false", "incorrect"],
        "HYPERNYM": ["truth-value property"],
        "DERIVATIONALLY_RELATED": ["truth"],
    },
    "may": {
        "pos": "AUX",
        "senses": [
            ("may.aux.01", "modal expressing possibility or permission")
        ],
        "SYNONYM": ["might", "could"],
        "ANTONYM": ["must not"],
        "COORDINATE_TERM": ["must", "can", "should"],
    },
    "not": {
        "pos": "PART",
        "senses": [
            ("not.part.01", "negation operator")
        ],
        "ANTONYM": ["affirmation"],
        "HYPERNYM": ["logical operator"],
    },
    "yet": {
        "pos": "ADV",
        "senses": [
            ("yet.adv.01", "up to the present or specified time"),
            ("yet.conj.01", "nevertheless; contrastive connective")
        ],
        "SYNONYM": ["still", "nevertheless"],
        "COORDINATE_TERM": ["already", "still"],
    },
    "whether": {
        "pos": "SCONJ",
        "senses": [
            ("whether.sconj.01", "introduces an embedded polar alternative or question")
        ],
        "COORDINATE_TERM": ["if"],
    },
    "love": {
        "pos": "VERB",
        "senses": [
            ("love.v.01", "feel strong affection or attachment toward a target"),
            ("love.n.01", "a strong positive affective attachment")
        ],
        "SYNONYM": ["adore", "cherish", "care for"],
        "ANTONYM": ["hate", "detest"],
        "HYPERNYM": ["affective attitude"],
        "DERIVATIONALLY_RELATED": ["loving", "lover", "lovable"],
        "COORDINATE_TERM": ["like", "admire", "trust", "desire"],
    },
    "hate": {
        "pos": "VERB",
        "senses": [
            ("hate.v.01", "feel intense dislike or hostility toward a target"),
            ("hate.n.01", "a strong negative affective attitude")
        ],
        "SYNONYM": ["detest", "loathe", "despise"],
        "ANTONYM": ["love", "adore"],
        "HYPERNYM": ["affective attitude"],
        "DERIVATIONALLY_RELATED": ["hatred", "hateful"],
        "COORDINATE_TERM": ["dislike", "resent", "fear", "hostility"],
    },
    "myself": {
        "pos": "PRON",
        "senses": [
            ("myself.pron.01", "first-person singular reflexive pronoun coreferential with the speaker")
        ],
        "COORDINATE_TERM": ["yourself", "himself", "herself", "ourselves", "themselves"],
    },
}

ENGLISH_PRONOUNS = {
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","mine","yours","hers","ours","theirs",
    "myself","yourself","yourselves","himself","herself","itself",
    "ourselves","themselves"
}
ENGLISH_DETERMINERS = {
    "a","an","the","this","that","these","those","some","any","each","every",
    "no","many","few","several","much"
}
ENGLISH_AUX = {
    "am","is","are","was","were","be","been","being",
    "do","does","did","have","has","had",
    "can","could","may","might","must","shall","should","will","would"
}
ENGLISH_CONJ = {
    "and","or","but","although","though","because","if","whether","while",
    "whereas","however","therefore","yet"
}
ENGLISH_COMMON_VERBS = {
    "believe","think","know","understand","support","conclude","infer","appear",
    "change","observe","measure","show","mean","represent","interpret","learn",
    "explain","cause","result","seem","suggest","indicate","prove","doubt",
    "love","hate","like","dislike","adore","cherish","detest","loathe","despise"
}
ENGLISH_COMMON_NOUNS = {
    "model","evidence","conclusion","result","information","claim","premise",
    "hypothesis","truth","knowledge","belief","reason","argument","data"
}
ENGLISH_COMMON_ADJ = {
    "true","false","sufficient","possible","certain","uncertain","correct",
    "incorrect","valid","invalid","new","old"
}

def _bootstrap_wordnet():
    """
    Attempt one quiet local bootstrap when NLTK is installed but its corpora
    are missing. Failure is harmless; ATLAS then uses curated fallback entries.
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("test")
            return True
        except Exception:
            pass
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            wn.synsets("test")
            return True
        except Exception:
            return False
    except Exception:
        return False

def _wordnet_resource():
    """Return WordNet if available, attempting one quiet bootstrap first."""
    try:
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("test")
            return wn
        except Exception:
            if _bootstrap_wordnet():
                return wn
            return None
    except Exception:
        return None

def _wn_pos(pos: str) -> str:
    return {"n":"NOUN","v":"VERB","a":"ADJ","s":"ADJ","r":"ADV"}.get(pos, "X")

def _lemma_text(name: str) -> str:
    return name.replace("_", " ")

def _append_lex_edge(edges, seen, source, target, relation,
                     source_sense="", target_sense="", weight=.5,
                     evidence="", provenance="ATLAS_LEXICAL_ENGINE_V10"):
    target = str(target).strip()
    if not target or target.casefold() == source.casefold():
        return
    key = (source.casefold(), target.casefold(), relation, source_sense, target_sense)
    if key in seen:
        return
    seen.add(key)
    edges.append(LexicalEdge(
        source=source, target=target, relation=relation,
        source_sense=source_sense, target_sense=target_sense,
        weight=clamp(weight), evidence=evidence,
        provenance=provenance
    ))

def build_lexical_neighborhood(tok: Token, max_targets: int = 18) -> LexicalNeighborhood:
    """
    Build a typed lexical neighborhood.

    English WordNet is used when locally available. Other languages are attempted
    through Open Multilingual WordNet if that corpus is installed. Unsupported
    relations are reported as missing evidence rather than assigned fake values.
    """
    all_types = list(LEXICAL_RELATION_REGISTRY)
    if tok.pos == "PUNCT":
        return LexicalNeighborhood(
            tok.id, tok.surface, tok.lemma, tok.language, [], [], all_types,
            {"resource":"none","reason":"punctuation"}, "NOT_APPLICABLE"
        )

    wn = _wordnet_resource()
    senses, edges, seen = [], [], set()
    diagnostics = {
        "wordnet_available": bool(wn),
        "requested_language": tok.language,
        "dictionary_backed": False,
        "homophone_resource": False,
    }

    if wn is None:
        entry = CURATED_LEXICON_V7.get(tok.lemma)
        if entry:
            for sid, definition in entry.get("senses", []):
                senses.append(LexicalSense(
                    id=sid,
                    lemma=tok.lemma,
                    pos=entry.get("pos", tok.pos),
                    definition=definition,
                    examples=[],
                    source="CURATED_FALLBACK_V7",
                    status="CURATED"
                ))
            for relation, targets in entry.items():
                if relation not in LEXICAL_RELATION_REGISTRY:
                    continue
                for target in targets:
                    _append_lex_edge(
                        edges, seen, tok.surface, target, relation,
                        weight=.70,
                        evidence="curated fallback lexical registry",
                        provenance="CURATED_FALLBACK_V7"
                    )
            present = {e.relation for e in edges}
            missing = [r for r in all_types if r not in present]
            return LexicalNeighborhood(
                tok.id, tok.surface, tok.lemma, tok.language,
                senses, edges, missing,
                {
                    **diagnostics,
                    "reason":"WordNet unavailable; curated fallback used",
                    "dictionary_backed":False,
                    "fallback":True,
                    "rule":"fallback evidence remains separately provenanced"
                },
                "CURATED_FALLBACK"
            )
        return LexicalNeighborhood(
            tok.id, tok.surface, tok.lemma, tok.language, senses, edges, all_types,
            {
                **diagnostics,
                "reason":"WordNet unavailable and no curated fallback entry",
                "rule":"absence of resource is UNKNOWN, not zero relations"
            },
            "RESOURCE_UNAVAILABLE"
        )

    try:
        synsets = wn.synsets(tok.lemma) if tok.language == "en" else wn.synsets(tok.lemma, lang=tok.language)
    except Exception:
        synsets = []

    diagnostics["dictionary_backed"] = bool(synsets)
    diagnostics["sense_count"] = len(synsets)

    if not synsets and tok.lemma in CURATED_LEXICON_V7:
        entry = CURATED_LEXICON_V7[tok.lemma]
        for sid, definition in entry.get("senses", []):
            senses.append(LexicalSense(
                id=sid, lemma=tok.lemma, pos=entry.get("pos", tok.pos),
                definition=definition, examples=[],
                source="CURATED_FALLBACK_V7", status="CURATED"
            ))
        for relation, targets in entry.items():
            if relation not in LEXICAL_RELATION_REGISTRY:
                continue
            for target in targets:
                _append_lex_edge(
                    edges, seen, tok.surface, target, relation,
                    weight=.70,
                    evidence="curated fallback lexical registry",
                    provenance="CURATED_FALLBACK_V7"
                )

    for syn in synsets[:32]:
        sid = syn.name()
        senses.append(LexicalSense(
            id=sid,
            lemma=tok.lemma,
            pos=_wn_pos(syn.pos()),
            definition=syn.definition(),
            examples=list(syn.examples())[:3],
            source="WORDNET/OMW"
        ))

        for lem in syn.lemmas()[:max_targets]:
            _append_lex_edge(
                edges, seen, tok.surface, _lemma_text(lem.name()),
                "SYNONYM", sid, sid, .92, syn.definition(), "WORDNET/OMW"
            )
            for ant in lem.antonyms()[:max_targets]:
                _append_lex_edge(
                    edges, seen, tok.surface, _lemma_text(ant.name()),
                    "ANTONYM", sid, ant.synset().name(), .96,
                    "WordNet antonym relation", "WORDNET/OMW"
                )
            for dr in lem.derivationally_related_forms()[:max_targets]:
                _append_lex_edge(
                    edges, seen, tok.surface, _lemma_text(dr.name()),
                    "DERIVATIONALLY_RELATED", sid, dr.synset().name(), .82,
                    "WordNet derivational relation", "WORDNET/OMW"
                )
            for p in lem.pertainyms()[:max_targets]:
                _append_lex_edge(
                    edges, seen, tok.surface, _lemma_text(p.name()),
                    "PERTAINYM", sid, p.synset().name(), .76,
                    "WordNet pertainym relation", "WORDNET/OMW"
                )

        def add_synsets(relation, targets, weight):
            count = 0
            for target_syn in targets:
                for name in target_syn.lemma_names()[:5]:
                    _append_lex_edge(
                        edges, seen, tok.surface, _lemma_text(name),
                        relation, sid, target_syn.name(), weight,
                        target_syn.definition(), "WORDNET/OMW"
                    )
                    count += 1
                    if count >= max_targets:
                        return

        add_synsets("HYPERNYM", syn.hypernyms(), .88)
        add_synsets("HYPONYM", syn.hyponyms(), .84)
        add_synsets("MERONYM_PART", syn.part_meronyms(), .82)
        add_synsets("MERONYM_MEMBER", syn.member_meronyms(), .82)
        add_synsets("MERONYM_SUBSTANCE", syn.substance_meronyms(), .82)
        add_synsets("HOLONYM_PART", syn.part_holonyms(), .82)
        add_synsets("HOLONYM_MEMBER", syn.member_holonyms(), .82)
        add_synsets("HOLONYM_SUBSTANCE", syn.substance_holonyms(), .82)
        add_synsets("ATTRIBUTE", syn.attributes(), .75)
        add_synsets("ENTAILMENT", syn.entailments(), .88)
        add_synsets("CAUSE", syn.causes(), .88)

        siblings = []
        for h in syn.hypernyms():
            siblings.extend([x for x in h.hyponyms() if x != syn])
        add_synsets("COORDINATE_TERM", siblings, .68)

        if syn.pos() == "v":
            add_synsets("TROPONYM", syn.hyponyms(), .78)

    # Sense multiplicity is represented explicitly. Homonymy is only a candidate:
    # multiple WordNet senses can be polysemy rather than true homonymy.
    if len(senses) > 1:
        for s in senses[:max_targets]:
            _append_lex_edge(
                edges, seen, tok.surface, s.id, "POLYSEMOUS_SENSE",
                target_sense=s.id, weight=.72, evidence=s.definition,
                provenance="ATLAS+WORDNET"
            )
        pos_types = {s.pos for s in senses}
        if len(pos_types) > 1:
            _append_lex_edge(
                edges, seen, tok.surface,
                f"{tok.surface} [same spelling, POS={','.join(sorted(pos_types))}]",
                "HOMOGRAPH_CANDIDATE", weight=.72,
                evidence="same written form appears across lexical categories",
                provenance="ATLAS+WORDNET"
            )
        _append_lex_edge(
            edges, seen, tok.surface,
            f"{tok.surface} [sense-cluster separation test]",
            "HOMONYM_CANDIDATE",
            weight=min(.85, .45 + .04*len(senses)),
            evidence=f"{len(senses)} dictionary senses; requires semantic-distance/etymology validation",
            provenance="ATLAS+WORDNET"
        )

    present = {e.relation for e in edges}
    missing = [r for r in all_types if r not in present]
    diagnostics["edge_count"] = len(edges)
    diagnostics["present_relation_types"] = sorted(present)
    diagnostics["important_limit"] = (
        "HOMOPHONE requires pronunciation evidence; COGNATE/FALSE_FRIEND require "
        "historical cross-language resources; semantic-neighbor/collocation require corpus observations."
    )

    return LexicalNeighborhood(
        token_id=tok.id,
        surface=tok.surface,
        lemma=tok.lemma,
        language=tok.language,
        senses=senses,
        edges=edges,
        missing_relation_types=missing,
        diagnostics=diagnostics,
        status="PROVISIONAL" if synsets else "NO_DICTIONARY_ENTRY"
    )

def lexical_edges_df(state: UtteranceState) -> pd.DataFrame:
    rows = []
    for ts in state.tokens:
        ln = ts.lexicalNeighborhood
        if not ln:
            continue
        for e in ln.edges:
            rows.append({
                "token": ts.token.surface,
                "lemma": ts.token.lemma,
                "relation": e.relation,
                "target": e.target,
                "source_sense": e.source_sense,
                "target_sense": e.target_sense,
                "weight": e.weight,
                "evidence": e.evidence,
                "provenance": e.provenance,
                "status": e.status,
            })
    return pd.DataFrame(rows)

def lexical_senses_df(state: UtteranceState) -> pd.DataFrame:
    rows = []
    for ts in state.tokens:
        ln = ts.lexicalNeighborhood
        if not ln:
            continue
        for s in ln.senses:
            rows.append({
                "token": ts.token.surface,
                "sense_id": s.id,
                "lemma": s.lemma,
                "pos": s.pos,
                "definition": s.definition,
                "examples": " | ".join(s.examples),
                "source": s.source,
                "status": s.status,
            })
    return pd.DataFrame(rows)

def lexical_coverage_df(state: UtteranceState) -> pd.DataFrame:
    rows = []
    for ts in state.tokens:
        if ts.token.pos == "PUNCT":
            continue
        ln = ts.lexicalNeighborhood
        present = sorted({e.relation for e in ln.edges}) if ln else []
        missing = ln.missing_relation_types if ln else list(LEXICAL_RELATION_REGISTRY)
        rows.append({
            "token": ts.token.surface,
            "lemma": ts.token.lemma,
            "senses": len(ln.senses) if ln else 0,
            "relations": len(ln.edges) if ln else 0,
            "present_types": ", ".join(present),
            "missing_or_unobserved_types": ", ".join(missing),
            "status": ln.status if ln else "UNAVAILABLE",
        })
    return pd.DataFrame(rows)

def lexical_graph(state: UtteranceState, max_edges_per_token: int = 30) -> go.Figure:
    G = nx.MultiDiGraph()
    for ts in state.tokens:
        if ts.token.pos == "PUNCT":
            continue
        root = f"T::{ts.token.id}"
        G.add_node(root, label=ts.token.surface, kind="TOKEN")
        ln = ts.lexicalNeighborhood
        if not ln:
            continue
        for e in sorted(ln.edges, key=lambda x: x.weight, reverse=True)[:max_edges_per_token]:
            target = f"{e.relation}::{e.target}"
            G.add_node(target, label=e.target, kind=e.relation)
            G.add_edge(root, target, relation=e.relation, weight=e.weight)

    fig = go.Figure()
    if not G.nodes:
        fig.update_layout(template="plotly_dark", height=500, title="No lexical relation data")
        return fig

    pos = nx.spring_layout(G, seed=42)
    ex, ey = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1),
                             hoverinfo="skip",name="typed lexical relation"))

    nxv, nyv, labels, hover = [], [], [], []
    for n,d in G.nodes(data=True):
        x,y = pos[n]
        nxv.append(x); nyv.append(y)
        labels.append(d.get("label",n))
        hover.append(f"<b>{d.get('label',n)}</b><br>{d.get('kind','')}")
    fig.add_trace(go.Scatter(
        x=nxv,y=nyv,mode="markers+text",text=labels,textposition="top center",
        customdata=hover,hovertemplate="%{customdata}<extra></extra>",
        marker=dict(size=9),name="lexical nodes"
    ))
    fig.update_layout(
        template="plotly_dark",height=760,title="ATLAS typed lexical neighborhood",
        showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),
        margin=dict(l=15,r=15,t=55,b=15)
    )
    return fig

# =============================================================================
# Uncertainty / validation / aggregation
# =============================================================================

def make_uncertainty(typed_spaces: dict[str, dict[str, float]]) -> UncertaintyState:
    vec = {}
    for s, coords in typed_spaces.items():
        if s in {"UNC","PHON","SOC","SYN","SCOPE"}:
            vec[s] = mean(coords.values())
    vals = list(vec.values()) or [0.5]
    n = len(vals)
    diag = np.clip(np.asarray(vals), 0.05, 0.95) ** 2
    cov = np.diag(diag)
    return UncertaintyState(
        vector={k: float(v) for k, v in vec.items()},
        covariance=cov.tolist(),
        entropy_proxy=simple_entropy(vals),
        ambiguity_preserved=True,
    )

def make_validation(typed_spaces: dict[str, dict[str, float]], uncertainty: UncertaintyState) -> ValidationState:
    all_vals = [v for coords in typed_spaces.values() for v in coords.values()]
    in_bounds = all(0 <= float(v) <= 1 for v in all_vals)
    consistency = clamp(1 - np.std(all_vals) * 0.25) if all_vals else 0.5
    confidence = clamp(0.70 - mean(uncertainty.vector.values(), 0.5) * 0.25)
    return ValidationState(
        admissibility="OK" if in_bounds else "FAIL",
        consistency=consistency,
        confidence=confidence,
        validationStatus="PROVISIONAL",
    )

def aggregate_spaces(token_states: list[TokenState]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for space in SPACE_ORDER:
        states = [ts.typed_spaces[space] for ts in token_states if space in ts.typed_spaces]
        keys = sorted({k for s in states for k in s})
        out[space] = {}
        for k in keys:
            vals = [s[k] for s in states if k in s]
            if k in {
                "negation","operator_strength","scope_operator","force_question",
                "cause","time_salience","contrastive_focus","meta_salience",
                "connector_strength"
            }:
                out[space][k] = max(vals) if vals else 0.0
            else:
                out[space][k] = mean(vals)
    return out


# =============================================================================
# Relations / hyperrelations / interpretations / transformations
# =============================================================================

def relation_extract(tokens: list[Token], spaces_by_token: list[dict[str, dict[str, float]]]) -> list[Relation]:
    """
    Provisional structure-aware English relation extraction.

    This is intentionally more conservative than v6:
      * adjacency is preserved only as SURFACE_NEXT, not treated as syntax
      * negation seeks the following lexical predicate, skipping adverbs/connectives
      * modal scope seeks a following predicate
      * epistemic predicates take proposition spans rather than the next token
      * evidence is linked to nearby support/claim structure, not punctuation/connectives
      * contrast markers connect clause heads
      * whether introduces an embedded proposition/query frame
    """
    rels: list[Relation] = []
    content = [t for t in tokens if t.pos != "PUNCT"]

    for a, b in zip(content, content[1:]):
        rels.append(Relation(
            a.id, b.id, "SURFACE_NEXT", "OBS/SYN", 0.20,
            evidence="surface adjacency only"
        ))

    def next_matching(i, allowed=None, skip=None):
        skip = skip or {"PUNCT","DET","PRON","ADV","SCONJ","PART"}
        for t in tokens[i+1:]:
            if t.pos == "PUNCT":
                continue
            if allowed and t.pos in allowed:
                return t
            if not allowed and t.pos not in skip:
                return t
        return None

    def prev_matching(i, allowed=None, skip=None):
        skip = skip or {"PUNCT","DET","ADV","SCONJ"}
        for t in reversed(tokens[:i]):
            if t.pos == "PUNCT":
                continue
            if allowed and t.pos in allowed:
                return t
            if not allowed and t.pos not in skip:
                return t
        return None

    # Candidate lexical predicates and clause heads.
    predicate_positions = [
        (i,t) for i,t in enumerate(tokens)
        if t.pos in {"VERB","AUX","ADJ"} and t.pos != "PUNCT"
    ]

    for i,t in enumerate(tokens):
        w = t.lemma

        if t.pos == "PRON":
            pred = next_matching(i, allowed={"VERB","AUX","ADJ"})
            if pred:
                rels.append(Relation(
                    t.id, pred.id, "SUBJECT_CANDIDATE", "SYN/SEM", .72,
                    evidence="pronoun preceding predicate"
                ))

        if t.pos == "DET":
            noun = next_matching(i, allowed={"NOUN","X"})
            if noun:
                rels.append(Relation(
                    t.id, noun.id, "DETERMINES", "SYN", .78,
                    evidence="determiner before nominal candidate"
                ))

        if has(w, NEG):
            pred = next_matching(i, allowed={"VERB","AUX","ADJ"})
            if pred:
                rels.append(Relation(
                    t.id, pred.id, "NEGATES_OR_SCOPES", "SCOPE/LOG", .94,
                    evidence="negation operator scoped to following predicate"
                ))

        if has(w, MODAL) or t.pos == "AUX" and w in {"may","might","could","can","must","should","would","will"}:
            pred = next_matching(i, allowed={"VERB","ADJ"})
            if pred:
                rels.append(Relation(
                    t.id, pred.id, "MODAL_SCOPES", "SCOPE/MOD", .90,
                    evidence="modal auxiliary governing following predicate"
                ))

        if has(w, BELIEF) or w in {"believe","think","suspect","assume","know","doubt"}:
            # Link the attitude predicate to the first meaningful embedded predicate.
            embedded = next_matching(i, allowed={"VERB","AUX","ADJ"})
            if embedded:
                rels.append(Relation(
                    t.id, embedded.id, "TAKES_PROPOSITION", "EPI/SYN", .88,
                    evidence="propositional-attitude predicate with embedded predicate"
                ))

        if w == "evidence":
            # Prefer support predicate to right; otherwise nearby proposition head.
            support = next((x for x in tokens[i+1:] if x.lemma in {"support","supports","indicate","indicates","show","shows","prove","proves"}), None)
            if support:
                rels.append(Relation(
                    t.id, support.id, "EVIDENCE_FOR", "EPI/ARG", .90,
                    evidence="evidence nominal linked to support predicate"
                ))
            else:
                head = prev_matching(i, allowed={"VERB","ADJ"})
                if head:
                    rels.append(Relation(
                        t.id, head.id, "EVIDENTIAL_ARGUMENT_OF", "EPI/ARG", .72,
                        evidence="evidence nominal associated with nearby predicate"
                    ))

        if w in {"but","although","however","yet"} and t.pos == "SCONJ":
            left = prev_matching(i, allowed={"VERB","ADJ","AUX"})
            right = next_matching(i, allowed={"VERB","ADJ","AUX"})
            if left and right:
                rels.append(Relation(
                    left.id, right.id, "CONTRASTS_WITH", "DISC", .86,
                    evidence=f"contrast connective: {w}"
                ))

        if w in {"because","therefore","so"}:
            left = prev_matching(i, allowed={"VERB","ADJ","AUX"})
            right = next_matching(i, allowed={"VERB","ADJ","AUX"})
            if left and right:
                rels.append(Relation(
                    left.id, right.id, "CAUSE_OR_EXPLANATION", "CAUS/DISC", .84,
                    evidence=f"causal/explanatory connective: {w}"
                ))

        if w == "yet":
            pred = next_matching(i, allowed={"VERB","ADJ","AUX"})
            if pred:
                rels.append(Relation(
                    t.id, pred.id, "TEMPORAL_OPERATOR", "TEMP", .84,
                    evidence="temporal yet modifies following predicate"
                ))

        if w == "whether":
            pred = next_matching(i, allowed={"VERB","ADJ","AUX"})
            if pred:
                rels.append(Relation(
                    t.id, pred.id, "INTRODUCES_POLAR_ALTERNATIVE", "SCOPE/LOG", .92,
                    evidence="whether-clause marker"
                ))

    # Basic predicate-object links for English benchmark patterns.
    for i,t in enumerate(tokens):
        if t.pos == "VERB":
            obj = next_matching(i, allowed={"NOUN","X","PRON"})
            if obj:
                rels.append(Relation(
                    obj.id, t.id, "ARGUMENT_OF", "SEM/SYN", .66,
                    evidence="nearest post-verbal nominal candidate"
                ))

    uniq = {}
    for r in rels:
        uniq[(r.source, r.target, r.relation, r.space)] = r
    return list(uniq.values())

def hyperrelation_extract(tokens: list[Token], rels: list[Relation]) -> list[Hyperrelation]:
    hypers = []
    lookup = {t.lemma:t for t in tokens}

    stance = next((t for t in tokens if t.lemma in {"believe","think","suspect","assume","know","doubt"}), None)
    evidence = next((t for t in tokens if t.lemma == "evidence"), None)
    modal = next((t for t in tokens if t.lemma in {"may","might","could","must","should"}), None)
    neg = next((t for t in tokens if t.lemma in {"not","never"}), None)
    whether = next((t for t in tokens if t.lemma == "whether"), None)
    conclusion = next((t for t in tokens if t.lemma in {"conclusion","claim","hypothesis"}), None)
    truth = next((t for t in tokens if t.lemma in {"true","false"}), None)

    if stance:
        nodes = [stance.id]
        roles = {"attitude": stance.id}
        if modal:
            nodes.append(modal.id); roles["modal"] = modal.id
        if evidence:
            nodes.append(evidence.id); roles["evidence"] = evidence.id
        hypers.append(Hyperrelation(
            id=uid("h"),
            relation="EPISTEMIC_PROPOSITION_FRAME",
            nodes=nodes,
            roles=roles,
            confidence=.76,
            status="PROVISIONAL"
        ))

    if neg and stance and stance.index_u > neg.index_u:
        hypers.append(Hyperrelation(
            id=uid("h"),
            relation="NEGATED_EPISTEMIC_STATE",
            nodes=[neg.id, stance.id],
            roles={"negation":neg.id,"epistemic_predicate":stance.id},
            confidence=.84,
            status="PROVISIONAL"
        ))

    if whether and truth:
        nodes = [whether.id, truth.id]
        roles = {"alternative_marker":whether.id, "truth_predicate":truth.id}
        if conclusion:
            nodes.append(conclusion.id); roles["proposition_subject"] = conclusion.id
        hypers.append(Hyperrelation(
            id=uid("h"),
            relation="POLAR_ALTERNATIVE_FRAME",
            nodes=nodes,
            roles=roles,
            confidence=.86,
            status="PROVISIONAL"
        ))

    if len(tokens) >= 3:
        hypers.append(Hyperrelation(
            id=uid("h"),
            relation="FORM_SENSE_CONTEXT_COUPLING",
            nodes=[t.id for t in tokens[:min(5,len(tokens))]],
            roles={"utterance_context":tokens[0].context_u},
            confidence=.55,
            status="HYPOTHESIZED"
        ))
    return hypers

def candidate_interpretations(text: str, typed: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    base = {
        "id": "I0",
        "label": "surface-compositional",
        "weight": 0.62,
        "status": "PROVISIONAL",
    }
    alt = {
        "id": "I1",
        "label": "context-sensitive alternative",
        "weight": 0.25,
        "status": "UNRESOLVED",
    }
    scope = {
        "id": "I2",
        "label": "scope/attachment alternative",
        "weight": 0.13,
        "status": "UNRESOLVED",
    }
    if typed.get("SCOPE", {}).get("scope_ambiguity", 0) > 0.4:
        scope["weight"] = 0.25
        base["weight"] = 0.52
        alt["weight"] = 0.23
    return [base, alt, scope]

def infer_transformations(typed: dict[str, dict[str, float]]) -> list[Transformation]:
    t = []
    if typed.get("EPI", {}).get("commitment", 0.5) > 0.70:
        t.append(
            Transformation(
                id=uid("tr"),
                source="current_utterance",
                target="weakened_commitment_variant",
                operation="EPISTEMIC_WEAKEN",
                affected_spaces=["EPI","UNC","PRAG"],
                delta={"commitment": -0.25, "uncertainty": +0.20},
            )
        )
    if typed.get("LOG", {}).get("negation", 0) > 0.5:
        t.append(
            Transformation(
                id=uid("tr"),
                source="current_utterance",
                target="negation_removed_variant",
                operation="REMOVE_NEGATION",
                affected_spaces=["LOG","SCOPE","SEM"],
                delta={"negation": -1.0},
            )
        )
    return t



# =============================================================================
# Reference / semantic-role / meaning-frame engine
# =============================================================================

REFLEXIVE_TO_PERSON = {
    "myself": ("1","singular","SPEAKER"),
    "yourself": ("2","singular","ADDRESSEE"),
    "yourselves": ("2","plural","ADDRESSEE"),
    "himself": ("3","singular","OTHER"),
    "herself": ("3","singular","OTHER"),
    "itself": ("3","singular","OTHER"),
    "ourselves": ("1","plural","SPEAKER_GROUP"),
    "themselves": ("3","plural","OTHER"),
}

AFFECTIVE_PREDICATES = {
    "love": ("LOVE", 0.90, "POSITIVE"),
    "adore": ("LOVE", 0.94, "POSITIVE"),
    "cherish": ("LOVE", 0.88, "POSITIVE"),
    "like": ("LIKE", 0.72, "POSITIVE"),
    "hate": ("HATE", 0.10, "NEGATIVE"),
    "detest": ("HATE", 0.06, "NEGATIVE"),
    "loathe": ("HATE", 0.04, "NEGATIVE"),
    "despise": ("HATE", 0.08, "NEGATIVE"),
    "dislike": ("DISLIKE", 0.28, "NEGATIVE"),
}

def build_reference_entities(tokens: list[Token]) -> tuple[list[ReferenceEntity], list[CoreferenceEdge]]:
    refs: list[ReferenceEntity] = []
    coref: list[CoreferenceEdge] = []
    speaker_ref = None
    addressee_ref = None

    for t in tokens:
        w = t.lemma
        if w in {"i","me","my","mine","myself"}:
            if speaker_ref is None:
                speaker_ref = ReferenceEntity(
                    id="ref_speaker",
                    label="SPEAKER",
                    person="1",
                    number="singular",
                    discourse_role="SPEAKER",
                    token_ids=[],
                    confidence=.96
                )
                refs.append(speaker_ref)
            speaker_ref.token_ids.append(t.id)

        elif w in {"you","your","yours","yourself","yourselves"}:
            if addressee_ref is None:
                addressee_ref = ReferenceEntity(
                    id="ref_addressee",
                    label="ADDRESSEE",
                    person="2",
                    number="unknown",
                    discourse_role="ADDRESSEE",
                    token_ids=[],
                    confidence=.94
                )
                refs.append(addressee_ref)
            addressee_ref.token_ids.append(t.id)

        elif t.pos == "PRON":
            refs.append(ReferenceEntity(
                id=uid("ref"),
                label=t.surface,
                person="3",
                number="unknown",
                discourse_role="OTHER",
                token_ids=[t.id],
                confidence=.72
            ))

    # Reflexive coreference
    for t in tokens:
        w = t.lemma
        if w in REFLEXIVE_TO_PERSON:
            if w == "myself" and speaker_ref:
                antecedent = next((x for x in tokens if x.lemma == "i"), None)
                if antecedent:
                    coref.append(CoreferenceEdge(
                        source=t.id,
                        target=antecedent.id,
                        relation="REFLEXIVE_COREFERENCE",
                        confidence=.98,
                        evidence="English first-person singular reflexive pronoun"
                    ))
            elif w in {"yourself","yourselves"} and addressee_ref:
                antecedent = next((x for x in tokens if x.lemma == "you"), None)
                if antecedent:
                    coref.append(CoreferenceEdge(
                        source=t.id,
                        target=antecedent.id,
                        relation="REFLEXIVE_COREFERENCE",
                        confidence=.97,
                        evidence="English second-person reflexive pronoun"
                    ))

    return refs, coref

def _nearest_left(tokens, idx, predicate):
    for t in reversed(tokens[:idx]):
        if t.pos == "PUNCT":
            continue
        if predicate(t):
            return t
    return None

def _nearest_right(tokens, idx, predicate):
    for t in tokens[idx+1:]:
        if t.pos == "PUNCT":
            continue
        if predicate(t):
            return t
    return None

def build_semantic_roles_and_frames(tokens: list[Token],
                                    coref: list[CoreferenceEdge]) -> tuple[list[SemanticRoleBinding], list[MeaningFrame]]:
    roles: list[SemanticRoleBinding] = []
    frames: list[MeaningFrame] = []

    coref_map = {c.source:c.target for c in coref}

    for i,t in enumerate(tokens):
        w = t.lemma
        if t.pos != "VERB":
            continue

        subj = _nearest_left(tokens, i, lambda x: x.pos == "PRON")
        obj = _nearest_right(tokens, i, lambda x: x.pos in {"PRON","NOUN","X"})

        if subj:
            role_name = "EXPERIENCER" if w in AFFECTIVE_PREDICATES else "AGENT"
            roles.append(SemanticRoleBinding(
                predicate=t.id,
                role=role_name,
                filler=subj.id,
                confidence=.92 if w in AFFECTIVE_PREDICATES else .78,
                evidence="nearest pre-verbal pronoun + predicate frame"
            ))

        if obj:
            role_name = "TARGET" if w in AFFECTIVE_PREDICATES else "THEME"
            roles.append(SemanticRoleBinding(
                predicate=t.id,
                role=role_name,
                filler=obj.id,
                confidence=.92 if w in AFFECTIVE_PREDICATES else .72,
                evidence="nearest post-verbal referential candidate + predicate frame"
            ))

        if w in AFFECTIVE_PREDICATES:
            affect_type, valence, polarity_name = AFFECTIVE_PREDICATES[w]

            subj_label = "UNKNOWN"
            if subj:
                subj_label = "SPEAKER" if subj.lemma in {"i","me","myself"} else subj.surface

            obj_label = "UNKNOWN"
            direction = "UNRESOLVED"
            if obj:
                if obj.id in coref_map and coref_map[obj.id] == (subj.id if subj else None):
                    obj_label = f"SELF({subj_label})"
                    direction = "SELF_DIRECTED"
                elif obj.lemma == "myself" and subj and subj.lemma == "i":
                    obj_label = "SELF(SPEAKER)"
                    direction = "SELF_DIRECTED"
                elif obj.lemma in {"you","yourself","yourselves"}:
                    obj_label = "ADDRESSEE"
                    direction = "OTHER_DIRECTED"
                else:
                    obj_label = obj.surface
                    direction = "OTHER_DIRECTED"

            canonical = f"{affect_type}({subj_label}, {obj_label})"
            frames.append(MeaningFrame(
                id=uid("frame"),
                frame_type="AFFECTIVE_ATTITUDE",
                predicate=affect_type,
                predicate_token=t.id,
                roles=[r for r in roles if r.predicate == t.id],
                polarity="AFFIRMATIVE",
                modality="ASSERTED",
                affect_type=affect_type,
                affect_valence=valence,
                direction=direction,
                canonical_form=canonical,
                confidence=.94 if subj and obj else .72,
                status="PROVISIONAL"
            ))

        # Generic proposition frame for non-affective transitive predicates.
        elif subj or obj:
            canonical_roles = []
            if subj:
                canonical_roles.append(f"ARG0={subj.surface}")
            if obj:
                canonical_roles.append(f"ARG1={obj.surface}")
            frames.append(MeaningFrame(
                id=uid("frame"),
                frame_type="PREDICATE_ARGUMENT",
                predicate=w.upper(),
                predicate_token=t.id,
                roles=[r for r in roles if r.predicate == t.id],
                polarity="NEGATED" if any(x.lemma in {"not","never"} and x.index_u < t.index_u for x in tokens[max(0,i-2):i]) else "AFFIRMATIVE",
                modality="MODAL" if any(x.lemma in {"may","might","could","must","should"} and x.index_u < t.index_u for x in tokens[max(0,i-2):i]) else "ASSERTED",
                affect_type="",
                affect_valence=None,
                direction="",
                canonical_form=f"{w.upper()}({', '.join(canonical_roles)})",
                confidence=.70,
                status="PROVISIONAL"
            ))

    return roles, frames

def add_frame_relations(relations: list[Relation],
                        roles: list[SemanticRoleBinding],
                        coref: list[CoreferenceEdge]) -> list[Relation]:
    out = list(relations)
    for rb in roles:
        if rb.role == "EXPERIENCER":
            out.append(Relation(rb.filler, rb.predicate, "EXPERIENCER_OF", "ROLE/AFFECT", rb.confidence, evidence=rb.evidence))
        elif rb.role == "TARGET":
            out.append(Relation(rb.predicate, rb.filler, "AFFECT_TARGET", "ROLE/AFFECT", rb.confidence, evidence=rb.evidence))
        elif rb.role == "AGENT":
            out.append(Relation(rb.filler, rb.predicate, "AGENT_OF", "ROLE/SEM", rb.confidence, evidence=rb.evidence))
        elif rb.role == "THEME":
            out.append(Relation(rb.predicate, rb.filler, "THEME", "ROLE/SEM", rb.confidence, evidence=rb.evidence))
    for c in coref:
        out.append(Relation(c.source, c.target, c.relation, "REF", c.confidence, evidence=c.evidence))

    uniq = {}
    for r in out:
        uniq[(r.source,r.target,r.relation,r.space)] = r
    return list(uniq.values())

def build_comprehension_certificate(tokens: list[Token],
                                    roles: list[SemanticRoleBinding],
                                    frames: list[MeaningFrame],
                                    coref: list[CoreferenceEdge]) -> ComprehensionCertificate:
    predicates = [t for t in tokens if t.pos == "VERB"]
    has_pred = bool(predicates)
    has_subject = any(r.role in {"AGENT","EXPERIENCER"} for r in roles)
    has_object = any(r.role in {"THEME","TARGET"} for r in roles)
    reflexives = [t for t in tokens if t.lemma in REFLEXIVE_TO_PERSON]
    coref_ok = not reflexives or bool(coref)
    affect_frames = [f for f in frames if f.frame_type == "AFFECTIVE_ATTITUDE"]
    affect_target = not affect_frames or all(any(r.role=="TARGET" for r in f.roles) for f in affect_frames)

    speaker_present = any(t.lemma in {"i","me","my","mine","myself"} for t in tokens)
    addressee_needed = any(t.lemma in {"you","your","yours","yourself","yourselves"} for t in tokens)
    addressee_resolved = (not addressee_needed) or any(t.lemma in {"you","your","yours","yourself","yourselves"} for t in tokens)

    ambiguities = []
    unresolved = []
    if not has_pred:
        unresolved.append("predicate")
    if has_pred and not has_subject:
        unresolved.append("subject/experiencer")
    if has_pred and not has_object:
        unresolved.append("object/target")
    if reflexives and not coref_ok:
        unresolved.append("reflexive antecedent")
    if affect_frames and not affect_target:
        unresolved.append("affect target")
    if any(t.pos == "X" for t in tokens if t.pos != "PUNCT"):
        ambiguities.append("unresolved POS/sense candidates")

    resolved_count = 8 - len(unresolved)
    conf = clamp(0.45 + 0.06*resolved_count - 0.04*len(ambiguities))

    return ComprehensionCertificate(
        predicate_resolved=has_pred,
        subject_resolved=has_subject,
        object_resolved=has_object,
        coreference_resolved=coref_ok,
        affect_target_resolved=affect_target,
        speaker_resolved=speaker_present or not any(t.lemma=="i" for t in tokens),
        addressee_resolved=addressee_resolved,
        scope_resolved=True,
        ambiguities=ambiguities,
        unresolved_items=unresolved,
        confidence=conf,
        status="PROVISIONAL"
    )

def build_structural_attention(token_states: list[TokenState],
                               roles: list[SemanticRoleBinding],
                               frames: list[MeaningFrame]) -> dict[str, Any]:
    role_scores = {}
    for r in roles:
        role_scores[r.filler] = max(role_scores.get(r.filler,0), .78)
        role_scores[r.predicate] = max(role_scores.get(r.predicate,0), .92)

    frame_predicates = {f.predicate_token for f in frames}

    rows = []
    for ts in token_states:
        if ts.token.pos == "PUNCT":
            priority = 0.02
        else:
            base = ts.typed_spaces["ATT"]["priority"]
            priority = base
            if ts.token.id in frame_predicates:
                priority += .35
            if ts.token.id in role_scores:
                priority += .25 * role_scores[ts.token.id]
            if ts.token.pos in {"VERB"}:
                priority += .12
            if ts.token.pos in {"PRON","NOUN"}:
                priority += .06
            priority = clamp(priority)
        rows.append({"token":ts.token.surface,"priority":priority})

    rows = sorted(rows, key=lambda x:x["priority"], reverse=True)
    return {
        "top_tokens": rows[:7],
        "method": "structural_centrality+role+uncertainty",
        "punctuation_suppressed": True,
        "status": "PROVISIONAL"
    }

def build_state_delta(typed: dict[str, dict[str, float]],
                      frames: list[MeaningFrame]) -> StateDelta:
    affect = mean([abs((f.affect_valence or .5)-.5)*2 for f in frames if f.affect_valence is not None], 0.0)
    self_state = typed.get("SELF",{}).get("self_model_update_pressure",0.0)
    social = max(
        typed.get("SOC_REL",{}).get("interpersonal_target",0.0),
        typed.get("SOC_REL",{}).get("self_relational",0.0)
    )
    return StateDelta(
        knowledge=typed.get("KNOW",{}).get("updateability",0.0),
        belief=typed.get("EPI",{}).get("commitment",0.0),
        affect=affect,
        self_state=self_state,
        social_relation=social,
        pragmatic=typed.get("PRAG",{}).get("force_assertive",0.0),
        uncertainty=typed.get("UNC",{}).get("aggregate",0.0)
    )

def generate_counterfactual_neighbors(text: str,
                                      frames: list[MeaningFrame]) -> list[CounterfactualNeighbor]:
    out: list[CounterfactualNeighbor] = []
    if not frames:
        return out

    for f in frames:
        if f.frame_type != "AFFECTIVE_ATTITUDE":
            continue

        lower = text.casefold()
        pred = f.predicate.casefold()

        # Affect polarity transformation
        if pred == "love" and "love" in lower:
            out.append(CounterfactualNeighbor(
                text=re.sub(r"\blove\b","hate",text,flags=re.I),
                operation="AFFECT_POLARITY_FLIP",
                changed_dimensions=["AFFECT.valence","AFFECT.positive_affect","AFFECT.negative_affect","SOC_REL.affinity","SOC_REL.hostility"],
                expected_invariants=["EXPERIENCER","TARGET","tense","person"],
                expected_changes={"predicate":"LOVE→HATE","valence":"positive→negative"}
            ))
        elif pred == "hate" and "hate" in lower:
            out.append(CounterfactualNeighbor(
                text=re.sub(r"\bhate\b","love",text,flags=re.I),
                operation="AFFECT_POLARITY_FLIP",
                changed_dimensions=["AFFECT.valence","AFFECT.positive_affect","AFFECT.negative_affect","SOC_REL.affinity","SOC_REL.hostility"],
                expected_invariants=["EXPERIENCER","TARGET","tense","person"],
                expected_changes={"predicate":"HATE→LOVE","valence":"negative→positive"}
            ))

        # Target transformation
        if "myself" in lower:
            out.append(CounterfactualNeighbor(
                text=re.sub(r"\bmyself\b","you",text,flags=re.I),
                operation="TARGET_SELF_TO_OTHER",
                changed_dimensions=["REF.reflexive","SELF.self_target","SOC_REL.self_relational","SOC_REL.interpersonal_target"],
                expected_invariants=["predicate","EXPERIENCER","polarity"],
                expected_changes={"target":"SELF(SPEAKER)→ADDRESSEE"}
            ))
        elif re.search(r"\byou\b", lower):
            out.append(CounterfactualNeighbor(
                text=re.sub(r"\byou\b","myself",text,flags=re.I),
                operation="TARGET_OTHER_TO_SELF",
                changed_dimensions=["REF.reflexive","SELF.self_target","SOC_REL.self_relational","SOC_REL.interpersonal_target"],
                expected_invariants=["predicate","EXPERIENCER","polarity"],
                expected_changes={"target":"ADDRESSEE→SELF(SPEAKER)"}
            ))

        # Negation transformation (distinct from lexical antonymy)
        if " not " not in f" {lower} ":
            subj_match = re.match(r"^\s*(i)\s+", text, flags=re.I)
            if subj_match and pred in lower:
                neg_text = re.sub(r"^\s*i\s+", "I do not ", text, count=1, flags=re.I)
                out.append(CounterfactualNeighbor(
                    text=neg_text,
                    operation="GRAMMATICAL_NEGATION",
                    changed_dimensions=["LOG.negation","SCOPE.scope_operator","PRAG.force_assertive"],
                    expected_invariants=["lexical predicate","EXPERIENCER","TARGET"],
                    expected_changes={"logical_polarity":"affirmative→negative"}
                ))

    # deduplicate
    uniq = {}
    for n in out:
        uniq[(n.text,n.operation)] = n
    return list(uniq.values())


# =============================================================================
# Active native language maps
# =============================================================================

LANGUAGE_NAMES_BY_CODE = {code:name for name,code in LANGUAGES.items()}

# These are structural map specifications, not claims that every coordinate is
# exhaustively measured. Values are native-mechanics indicators used by the map
# and by map-to-map comparison. Missing coordinates remain absent, not zero.
LANGUAGE_NATIVE_SPECS: dict[str, dict[str, Any]] = {
    "en": {
        "script":"Latin",
        "mechanics":{
            "word_order":"SVO","adposition":"preposition","head_direction":"mixed_head_initial",
            "morphology":"mostly_analytic_fusional","pro_drop":False,
            "reflexive_strategy":"self_pronoun","object_marking":"position",
            "agreement":"limited","evidentiality":"lexical","topic_prominence":"moderate",
        },
        "axes":{
            "ORTH":{"latin_script":1.0,"word_spacing":1.0,"case_distinction":1.0},
            "MORPH":{"analyticity":0.78,"fusion":0.34,"agglutination":0.12,"templaticity":0.02},
            "SYN":{"svo":0.95,"sov":0.05,"vso":0.05,"head_final":0.28,"prepositions":0.96,"postpositions":0.08,"pro_drop":0.12},
            "REF":{"overt_pronoun_pressure":0.86,"reflexive_pronoun":0.96,"agreement_recovery":0.22,"clitic_reference":0.08},
            "SCOPE":{"auxiliary_modality":0.88,"preverbal_negation":0.92},
            "EPI":{"lexical_evidentiality":0.92,"grammatical_evidentiality":0.08},
            "INFO":{"topic_prominence":0.48,"scrambling":0.18,"focus_particles":0.18},
        },
        "operators":["DO_NEGATION","AUX_MODAL","TENSE_AUX","WH_MOVEMENT","PASSIVE_BE"],
        "constraints":["FINITE_CLAUSE_WORD_ORDER","REFLEXIVE_ANTECEDENT_AGREEMENT"],
    },
    "fa": {
        "script":"Perso-Arabic",
        "mechanics":{
            "word_order":"SOV","adposition":"mostly_prepositions+postpositional_ra",
            "head_direction":"mixed_head_final","morphology":"fusional_agglutinative_mix",
            "pro_drop":True,"reflexive_strategy":"xod+self_clitic",
            "object_marking":"differential_ra","agreement":"verbal_person_number",
            "ezafe":True,"light_verbs":True,"evidentiality":"lexical/aspectual",
            "topic_prominence":"high_scrambling",
        },
        "axes":{
            "ORTH":{"arabic_script":1.0,"short_vowel_omission":0.96,"joining_behavior":0.94,"zwnj_relevance":0.88},
            "MORPH":{"analyticity":0.34,"fusion":0.52,"agglutination":0.58,"cliticization":0.78,"ezafe":0.96,"light_verb_productivity":0.94},
            "SYN":{"svo":0.18,"sov":0.94,"vso":0.04,"head_final":0.78,"prepositions":0.82,"postpositions":0.34,"pro_drop":0.82,"differential_object_marker":0.96},
            "REF":{"overt_pronoun_pressure":0.36,"reflexive_pronoun":0.82,"agreement_recovery":0.82,"clitic_reference":0.76},
            "SCOPE":{"prefixal_negation":0.92,"modal_construction":0.72},
            "EPI":{"lexical_evidentiality":0.82,"grammatical_evidentiality":0.22},
            "INFO":{"topic_prominence":0.76,"scrambling":0.86,"focus_particles":0.42},
        },
        "operators":["NA_NEGATION","MI_IMPERFECTIVE","RA_OBJECT_MARKER","EZAFE_LINK","LIGHT_VERB_COMPOSE","PERSON_SUFFIX"],
        "constraints":["SOV_CANONICAL","RA_DIFFERENTIAL_OBJECT","EZAFE_NOMINAL_LINK","PRO_DROP_AGREEMENT_RECOVERY"],
    },
    "ar": {
        "script":"Arabic",
        "mechanics":{
            "word_order":"VSO/SVO","adposition":"preposition","head_direction":"mixed_head_initial",
            "morphology":"root_pattern_fusional","pro_drop":True,
            "reflexive_strategy":"nafs+self_possessive","object_marking":"case/clitic/position",
            "agreement":"rich_person_gender_number","root_pattern":True,
            "attached_pronouns":True,"evidentiality":"lexical",
        },
        "axes":{
            "ORTH":{"arabic_script":1.0,"short_vowel_omission":0.96,"joining_behavior":0.96},
            "MORPH":{"fusion":0.84,"agglutination":0.34,"templaticity":0.98,"cliticization":0.92,"root_pattern":0.98},
            "SYN":{"svo":0.72,"sov":0.08,"vso":0.84,"head_final":0.18,"prepositions":0.96,"postpositions":0.04,"pro_drop":0.78},
            "REF":{"overt_pronoun_pressure":0.38,"reflexive_pronoun":0.84,"agreement_recovery":0.88,"clitic_reference":0.94},
            "SCOPE":{"prefixal_negation":0.74,"particle_negation":0.92,"modal_particles":0.78},
            "EPI":{"lexical_evidentiality":0.88,"grammatical_evidentiality":0.12},
            "INFO":{"topic_prominence":0.58,"scrambling":0.58,"focus_particles":0.54},
        },
        "operators":["ROOT_PATTERN_DERIVE","NEG_PARTICLE","ATTACHED_PRONOUN","AGREEMENT_INFLECT","VSO_SVO_ALTERNATION"],
        "constraints":["ROOT_PATTERN_COMPATIBILITY","AGREEMENT_PERSON_GENDER_NUMBER","CLITIC_ATTACHMENT"],
    },
    "zh": {
        "script":"Han",
        "mechanics":{
            "word_order":"SVO","adposition":"coverb/preposition","head_direction":"mixed",
            "morphology":"isolating","pro_drop":True,"reflexive_strategy":"ziji/wo-ziji",
            "object_marking":"position/ba_construction","agreement":"minimal",
            "aspect_particles":True,"topic_prominence":"high","classifier_system":True,
            "evidentiality":"particles/lexical",
        },
        "axes":{
            "ORTH":{"han_script":1.0,"word_spacing":0.05,"logographic_pressure":0.96},
            "MORPH":{"analyticity":0.98,"fusion":0.04,"agglutination":0.06,"templaticity":0.01,"classifier_dependence":0.88},
            "SYN":{"svo":0.94,"sov":0.18,"vso":0.02,"head_final":0.44,"prepositions":0.68,"postpositions":0.24,"pro_drop":0.72,"topic_comment":0.94},
            "REF":{"overt_pronoun_pressure":0.48,"reflexive_pronoun":0.92,"agreement_recovery":0.08,"clitic_reference":0.04},
            "SCOPE":{"modal_verb":0.74,"negation_particles":0.94,"aspect_particles":0.92},
            "EPI":{"lexical_evidentiality":0.78,"grammatical_evidentiality":0.18},
            "INFO":{"topic_prominence":0.96,"scrambling":0.42,"focus_particles":0.78},
        },
        "operators":["LE_ASPECT","GUO_EXPERIENTIAL","ZAI_PROGRESSIVE","BU_NEGATION","MEI_NEGATION","BA_CONSTRUCTION","BEI_PASSIVE"],
        "constraints":["CLASSIFIER_NOUN_COMPATIBILITY","TOPIC_COMMENT_STRUCTURE","ASPECT_PARTICLE_POSITION"],
    },
    "tr": {
        "script":"Latin",
        "mechanics":{
            "word_order":"SOV","adposition":"postposition","head_direction":"head_final",
            "morphology":"agglutinative","pro_drop":True,
            "reflexive_strategy":"kendi+possessive+case",
            "object_marking":"accusative_specificity","agreement":"rich_person_number",
            "vowel_harmony":True,"evidentiality":"grammatical_direct/inferential",
        },
        "axes":{
            "ORTH":{"latin_script":1.0,"word_spacing":1.0,"case_distinction":1.0},
            "PHON":{"vowel_harmony":0.98,"agglutinative_phonology_interface":0.92},
            "MORPH":{"analyticity":0.12,"fusion":0.12,"agglutination":0.98,"case_suffixing":0.96,"possessive_suffixing":0.96,"person_suffixing":0.96},
            "SYN":{"svo":0.12,"sov":0.96,"vso":0.03,"head_final":0.94,"prepositions":0.04,"postpositions":0.96,"pro_drop":0.92},
            "REF":{"overt_pronoun_pressure":0.22,"reflexive_pronoun":0.96,"agreement_recovery":0.94,"clitic_reference":0.18},
            "SCOPE":{"suffixal_negation":0.96,"modal_suffixes":0.88},
            "EPI":{"lexical_evidentiality":0.62,"grammatical_evidentiality":0.96,"inferential_past":0.96},
            "INFO":{"topic_prominence":0.68,"scrambling":0.88,"focus_particles":0.46},
        },
        "operators":["VOWEL_HARMONY","CASE_SUFFIX","PERSON_SUFFIX","NEGATIVE_SUFFIX","EVIDENTIAL_MIS","PROGRESSIVE_IYOR"],
        "constraints":["VOWEL_HARMONY_COMPATIBILITY","SUFFIX_ORDER","CASE_ROLE_MAPPING","PRO_DROP_AGREEMENT_RECOVERY"],
    },
    "ja": {
        "script":"Kanji+Kana",
        "mechanics":{"word_order":"SOV","adposition":"postposition","head_direction":"head_final","morphology":"agglutinative","pro_drop":True,"topic_prominence":"high","honorific_system":True},
        "axes":{
            "ORTH":{"han_script":0.62,"kana_script":1.0,"word_spacing":0.05},
            "MORPH":{"agglutination":0.88,"fusion":0.18,"particle_marking":0.96},
            "SYN":{"sov":0.96,"head_final":0.96,"postpositions":0.98,"pro_drop":0.94},
            "REF":{"overt_pronoun_pressure":0.18,"agreement_recovery":0.18,"discourse_recovery":0.94},
            "INFO":{"topic_prominence":0.98,"focus_particles":0.88},
            "PRAG":{"honorific_encoding":0.96,"register_morphology":0.92},
        },
        "operators":["CASE_PARTICLE","TOPIC_WA","NEG_AUX","TENSE_AUX","HONORIFIC_REGISTER"],
        "constraints":["PARTICLE_ROLE_MAPPING","HEAD_FINAL_CLAUSE","DISCOURSE_ARGUMENT_RECOVERY"],
    },
    "ko": {
        "script":"Hangul",
        "mechanics":{"word_order":"SOV","adposition":"postposition","head_direction":"head_final","morphology":"agglutinative","pro_drop":True,"topic_prominence":"high","speech_levels":True},
        "axes":{
            "ORTH":{"hangul_script":1.0,"word_spacing":0.72},
            "MORPH":{"agglutination":0.94,"particle_marking":0.94,"verbal_ending_complexity":0.94},
            "SYN":{"sov":0.96,"head_final":0.96,"postpositions":0.96,"pro_drop":0.92},
            "REF":{"overt_pronoun_pressure":0.22,"discourse_recovery":0.92},
            "INFO":{"topic_prominence":0.94,"focus_particles":0.84},
            "PRAG":{"speech_level_encoding":0.98,"honorific_encoding":0.92},
        },
        "operators":["CASE_PARTICLE","TOPIC_PARTICLE","NEG_CONSTRUCTION","SPEECH_LEVEL_ENDING"],
        "constraints":["PARTICLE_ROLE_MAPPING","HEAD_FINAL_CLAUSE","SPEECH_LEVEL_AGREEMENT"],
    },
    "ru": {
        "script":"Cyrillic",
        "mechanics":{"word_order":"flexible_SVO","morphology":"fusional","pro_drop":False,"case_system":True,"aspect_pairs":True},
        "axes":{
            "ORTH":{"cyrillic_script":1.0,"word_spacing":1.0},
            "MORPH":{"fusion":0.94,"case_inflection":0.96,"aspectual_morphology":0.92},
            "SYN":{"svo":0.74,"scrambling":0.88,"pro_drop":0.18},
            "REF":{"agreement_recovery":0.72,"overt_pronoun_pressure":0.68},
            "TEMP":{"aspect_lexical_morphological":0.94},
            "INFO":{"scrambling":0.92,"topic_prominence":0.72},
        },
        "operators":["CASE_INFLECT","ASPECT_PAIR","NEG_PARTICLE","WORD_ORDER_INFORMATION_STRUCTURE"],
        "constraints":["CASE_ROLE_MAPPING","AGREEMENT_GENDER_NUMBER_PERSON"],
    },
    "he": {
        "script":"Hebrew",
        "mechanics":{"word_order":"SVO","morphology":"root_pattern_fusional","pro_drop":"partial","attached_pronouns":True},
        "axes":{
            "ORTH":{"hebrew_script":1.0,"short_vowel_omission":0.94},
            "MORPH":{"fusion":0.82,"templaticity":0.92,"cliticization":0.82},
            "SYN":{"svo":0.88,"pro_drop":0.52,"prepositions":0.94},
            "REF":{"agreement_recovery":0.82,"clitic_reference":0.82},
        },
        "operators":["ROOT_PATTERN_DERIVE","NEG_PARTICLE","AGREEMENT_INFLECT","CLITIC_ATTACH"],
        "constraints":["ROOT_PATTERN_COMPATIBILITY","AGREEMENT_FEATURES"],
    },
    "ur": {
        "script":"Perso-Arabic",
        "mechanics":{"word_order":"SOV","morphology":"fusional_analytic_mix","pro_drop":True,"postpositions":True,"gender":True},
        "axes":{
            "ORTH":{"arabic_script":1.0,"short_vowel_omission":0.94},
            "MORPH":{"fusion":0.68,"analyticity":0.44,"case_postposition_interface":0.88},
            "SYN":{"sov":0.96,"postpositions":0.98,"pro_drop":0.72},
            "REF":{"agreement_recovery":0.72,"overt_pronoun_pressure":0.38},
        },
        "operators":["POSTPOSITION_CASE","AUX_TAM","NEGATION","AGREEMENT"],
        "constraints":["SOV_CANONICAL","POSTPOSITION_ROLE_MAPPING"],
    },
    "la": {
        "script":"Latin",
        "mechanics":{"word_order":"flexible","morphology":"fusional","pro_drop":True,"case_system":True},
        "axes":{
            "ORTH":{"latin_script":1.0,"word_spacing":1.0},
            "MORPH":{"fusion":0.98,"case_inflection":0.98,"agreement":0.96},
            "SYN":{"scrambling":0.94,"pro_drop":0.92,"case_driven_roles":0.98},
            "REF":{"agreement_recovery":0.88,"overt_pronoun_pressure":0.18},
        },
        "operators":["CASE_INFLECT","AGREEMENT_INFLECT","TENSE_MOOD_INFLECT"],
        "constraints":["CASE_ROLE_MAPPING","AGREEMENT_FEATURES"],
    },
    "sa": {
        "script":"Devanagari / Brahmi-derived",
        "mechanics":{"word_order":"flexible_SOV","morphology":"fusional","pro_drop":True,"case_system":True,"sandhi":True},
        "axes":{
            "ORTH":{"devanagari_script":1.0},
            "PHON":{"sandhi":0.98},
            "MORPH":{"fusion":0.96,"case_inflection":0.98,"derivational_productivity":0.94},
            "SYN":{"sov":0.82,"scrambling":0.96,"pro_drop":0.88,"case_driven_roles":0.98},
        },
        "operators":["SANDHI","CASE_INFLECT","VERBAL_DERIVE","COMPOUND_FORM"],
        "constraints":["CASE_ROLE_MAPPING","SANDHI_ENVIRONMENT","AGREEMENT_FEATURES"],
    },
    "nv": {
        "script":"Latin",
        "mechanics":{"morphology":"polysynthetic","verb_centric":True,"word_order":"flexible","classifier_verbs":True},
        "axes":{
            "ORTH":{"latin_script":1.0},
            "MORPH":{"polysynthesis":0.98,"verbal_template":0.98,"incorporation_pressure":0.82},
            "SYN":{"verb_centric":0.98,"word_order_flexibility":0.82},
            "ROLE":{"role_in_verbal_morphology":0.94},
        },
        "operators":["VERBAL_TEMPLATE","CLASSIFIER_VERB","PERSON_PREFIX","ASPECT_MODE"],
        "constraints":["VERBAL_TEMPLATE_ORDER","CLASSIFIER_COMPATIBILITY"],
    },
    "apa": {
        "script":"Latin",
        "mechanics":{"morphology":"polysynthetic","verb_centric":True,"word_order":"flexible"},
        "axes":{
            "ORTH":{"latin_script":1.0},
            "MORPH":{"polysynthesis":0.98,"verbal_template":0.96},
            "SYN":{"verb_centric":0.96,"word_order_flexibility":0.82},
            "ROLE":{"role_in_verbal_morphology":0.92},
        },
        "operators":["VERBAL_TEMPLATE","PERSON_PREFIX","ASPECT_MODE"],
        "constraints":["VERBAL_TEMPLATE_ORDER"],
    },
    "chr": {
        "script":"Cherokee syllabary / Latin",
        "mechanics":{"morphology":"polysynthetic","verb_centric":True,"pronominal_prefixes":True},
        "axes":{
            "ORTH":{"syllabary_script":0.96},
            "MORPH":{"polysynthesis":0.94,"pronominal_prefixing":0.96},
            "SYN":{"verb_centric":0.92,"word_order_flexibility":0.78},
            "REF":{"reference_in_verbal_morphology":0.94},
        },
        "operators":["PRONOMINAL_PREFIX","VERBAL_STEM","ASPECT_SUFFIX"],
        "constraints":["PRONOMINAL_PREFIX_COMPATIBILITY","VERBAL_TEMPLATE_ORDER"],
    },
    "sux": {
        "script":"Cuneiform",
        "mechanics":{"word_order":"SOV","morphology":"agglutinative","case_system":True,"historical_corpus":True},
        "axes":{
            "ORTH":{"cuneiform_script":1.0},
            "MORPH":{"agglutination":0.92,"case_suffixing":0.92},
            "SYN":{"sov":0.94,"head_final":0.88,"postpositions":0.82},
        },
        "operators":["CASE_SUFFIX","VERBAL_PREFIX_CHAIN","POSSESSIVE_SUFFIX"],
        "constraints":["HISTORICAL_CORPUS_UNCERTAINTY","SUFFIX_ORDER"],
    },
    "und": {
        "script":"Unknown",
        "mechanics":{"status":"unknown"},
        "axes":{},
        "operators":[],
        "constraints":["DO_NOT_ASSUME_LANGUAGE_MECHANICS"],
    },
}

# Lexical/POS evidence used by the active analyzers. These are intentionally
# narrow and testable; unsupported material remains unresolved instead of being
# forced into English categories.
NATIVE_LEXICAL_POS = {
    "fa":{
        "من":"PRON","تو":"PRON","شما":"PRON","او":"PRON","خودم":"PRON","خودت":"PRON","خودش":"PRON",
        "را":"PART","از":"ADP","به":"ADP","که":"SCONJ","ممکن":"ADJ","شاید":"ADV",
        "دوست":"NOUN","دارم":"VERB","داری":"VERB","دارد":"VERB","متنفرم":"VERB","متنفری":"VERB",
        "باور":"NOUN","فکر":"NOUN","شواهد":"NOUN","مدرک":"NOUN","نتیجه":"NOUN","پشتیبانی":"NOUN",
        "کنند":"VERB","کند":"VERB","می‌کنم":"VERB","میکنم":"VERB","می‌کنم":"VERB",
    },
    "ar":{
        "أنا":"PRON","انا":"PRON","أنت":"PRON","انت":"PRON","أنتَ":"PRON","أنتِ":"PRON",
        "نفسي":"PRON","نفسك":"PRON","من":"ADP","في":"ADP","إلى":"ADP","أن":"SCONJ","أنّ":"SCONJ",
        "أحب":"VERB","احب":"VERB","أكره":"VERB","اكره":"VERB","أكرهك":"VERB","اكرهك":"VERB",
        "أعتقد":"VERB","اعتقد":"VERB","الدليل":"NOUN","الأدلة":"NOUN","ادلة":"NOUN","النتيجة":"NOUN",
        "قد":"PART","ربما":"ADV","تدعم":"VERB","يدعم":"VERB",
    },
    "zh":{
        "我":"PRON","你":"PRON","他":"PRON","她":"PRON","自己":"PRON","我自己":"PRON","你自己":"PRON",
        "爱":"VERB","愛":"VERB","恨":"VERB","喜欢":"VERB","喜歡":"VERB","认为":"VERB","認為":"VERB",
        "相信":"VERB","知道":"VERB","可能":"AUX","也许":"ADV","也許":"ADV","不":"PART","没":"PART","沒有":"PART",
        "证据":"NOUN","證據":"NOUN","结论":"NOUN","結論":"NOUN","支持":"VERB","这个":"DET","這個":"DET",
    },
    "tr":{
        "ben":"PRON","sen":"PRON","beni":"PRON","seni":"PRON","kendim":"PRON","kendimi":"PRON",
        "kendini":"PRON","senden":"PRON","benden":"PRON","kanıt":"NOUN","kanıtlar":"NOUN","kanıtların":"NOUN",
        "sonuç":"NOUN","sonucu":"NOUN","seviyorum":"VERB","seviyorum":"VERB","severim":"VERB",
        "nefret":"NOUN","ediyorum":"VERB","ediyorum":"VERB","inanıyorum":"VERB","inanıyorum":"VERB",
        "düşünüyorum":"VERB","destekleyebilir":"VERB","destekleyebileceğine":"VERB","mümkün":"ADJ","belki":"ADV",
    },
}

ZH_SEGMENT_LEXICON = sorted({
    "我自己","你自己","为什么","為什麼","证据","證據","结论","結論","认为","認為",
    "相信","知道","可能","支持","这个","這個","喜欢","喜歡","自己","我","你","他","她",
    "爱","愛","恨","不","没","沒有","也许","也許"
}, key=len, reverse=True)

def _base_token_from_surface(surface: str, start: int, end: int, idx: int, lang: str, utt_id: str,
                             pos: str | None = None, lemma: str | None = None, morph: str | None = None) -> Token:
    p = pos or guess_pos(surface)
    return Token(
        id=f"t_{idx}",
        index_u=idx,
        surface=surface,
        lemma=(lemma or surface.casefold()),
        pos=p,
        morph=morph if morph is not None else guess_morph(surface,p),
        span=(start,end),
        context_u=utt_id,
        language=lang,
    )

class AtlasLanguageMap:
    """Active computational language map.

    The map is not display metadata. It is the language-specific observation
    interface through which ATLAS tokenizes, analyzes, resolves reference and
    roles, constructs frames, and emits native typed coordinates.
    """
    analyzer_version = "ATLAS_NATIVE_MAP_V10"

    def __init__(self, lang: str):
        self.lang = lang
        self.name = LANGUAGE_NAMES_BY_CODE.get(lang,lang)
        self.spec = LANGUAGE_NATIVE_SPECS.get(lang, LANGUAGE_NATIVE_SPECS["und"])

    def tokenizer_name(self) -> str:
        return "unicode_word_punctuation"

    def tokenize(self, text: str, utt_id: str) -> list[Token]:
        out = []
        for m in TOKEN_RE.finditer(text):
            surface = m.group(0)
            if not surface.strip():
                continue
            pos = self.analyze_pos(surface)
            morph = self.analyze_morphology(surface,pos)
            out.append(_base_token_from_surface(surface,m.start(),m.end(),len(out),self.lang,utt_id,pos=pos,morph=morph))
        return out

    def analyze_pos(self, surface: str) -> str:
        # English keeps the richer existing recognizer.
        if self.lang == "en":
            return guess_pos(surface)
        entry = NATIVE_LEXICAL_POS.get(self.lang,{}).get(surface.casefold())
        if entry:
            return entry
        if len(surface)==1 and unicodedata.category(surface).startswith("P"):
            return "PUNCT"
        if has(surface,NEG):
            return "PART"
        if has(surface,MODAL):
            return "AUX"
        return "X"

    def analyze_morphology(self, surface: str, pos: str) -> str:
        if self.lang == "en":
            return guess_morph(surface,pos)
        return "_"

    def native_mechanics_observed(self, text: str, tokens: list[Token]) -> dict[str,Any]:
        return {
            "language":self.lang,
            "analyzer":self.__class__.__name__,
            "tokenizer":self.tokenizer_name(),
            "canonical_word_order":self.spec.get("mechanics",{}).get("word_order","unknown"),
            "morphology_type":self.spec.get("mechanics",{}).get("morphology","unknown"),
            "observed_operators":[op for op in self.spec.get("operators",[]) if self.operator_observed(op,text,tokens)],
            "status":"PROVISIONAL_NATIVE_ANALYSIS",
        }

    def operator_observed(self, operator: str, text: str, tokens: list[Token]) -> bool:
        # Base class conservatively reports no operator unless overridden.
        return False

    def encode_native_token_state(self, tok: Token, text: str) -> dict[str,dict[str,float]]:
        states = token_space_state(tok,text,self.lang)
        # Native map coordinates condition the token state without replacing
        # semantic state with map metadata.
        axes = self.spec.get("axes",{})
        for space in ("ORTH","PHON","MORPH","SYN","REF","SCOPE","EPI","INFO","PRAG","ROLE"):
            if space not in states:
                states[space] = {}
            for name,val in axes.get(space,{}).items():
                states[space][f"native::{name}"] = float(val)
        states.setdefault("META",{})["native_language_map_used"] = 1.0
        return states

    def resolve_reference(self, tokens: list[Token]) -> tuple[list[ReferenceEntity],list[CoreferenceEdge]]:
        if self.lang == "en":
            return build_reference_entities(tokens)
        refs = []
        for t in tokens:
            if t.pos == "PRON":
                refs.append(ReferenceEntity(
                    id=uid("ref"),label=t.surface,person="unknown",number="unknown",
                    discourse_role="REFERENT",token_ids=[t.id],confidence=.55
                ))
        return refs,[]

    def assign_roles_and_frames(self, tokens: list[Token], coref: list[CoreferenceEdge]) -> tuple[list[SemanticRoleBinding],list[MeaningFrame]]:
        if self.lang == "en":
            return build_semantic_roles_and_frames(tokens,coref)
        return self._generic_order_aware_roles(tokens,coref)

    def _generic_order_aware_roles(self,tokens,coref):
        roles=[]; frames=[]
        verbs=[t for t in tokens if t.pos=="VERB"]
        for pred in verbs:
            i=pred.index_u
            before=[t for t in tokens[:i] if t.pos in {"PRON","NOUN"}]
            after=[t for t in tokens[i+1:] if t.pos in {"PRON","NOUN"}]
            word_order=self.spec.get("mechanics",{}).get("word_order","")
            subj=(before[0] if before else None)
            obj=None
            if "SOV" in word_order:
                obj=(before[-1] if len(before)>=2 else (after[0] if after else None))
            else:
                obj=(after[0] if after else (before[-1] if len(before)>=2 else None))
            if subj:
                roles.append(SemanticRoleBinding(pred.id,"AGENT",subj.id,.60,"native-map order-aware role candidate"))
            if obj and (not subj or obj.id!=subj.id):
                roles.append(SemanticRoleBinding(pred.id,"THEME",obj.id,.58,"native-map order-aware role candidate"))
            if subj or obj:
                args=[]
                if subj: args.append(f"ARG0={subj.surface}")
                if obj: args.append(f"ARG1={obj.surface}")
                frames.append(MeaningFrame(
                    id=uid("frame"),frame_type="PREDICATE_ARGUMENT",
                    predicate=pred.lemma.upper(),predicate_token=pred.id,
                    roles=[r for r in roles if r.predicate==pred.id],
                    polarity="AFFIRMATIVE",modality="ASSERTED",
                    affect_type="",affect_valence=None,direction="",
                    canonical_form=f"{pred.lemma.upper()}({', '.join(args)})",
                    confidence=.60,status="PROVISIONAL_NATIVE"
                ))
        return roles,frames

    def extract_relations(self,tokens,spaces,roles,coref) -> list[Relation]:
        if self.lang=="en":
            rels=relation_extract(tokens,spaces)
        else:
            rels=[]
            content=[t for t in tokens if t.pos!="PUNCT"]
            for a,b in zip(content,content[1:]):
                rels.append(Relation(a.id,b.id,"SURFACE_NEXT","OBS",.16,evidence=f"{self.lang} surface adjacency only"))
        return add_frame_relations(rels,roles,coref)

    def extract_hyperrelations(self,tokens,rels,frames) -> list[Hyperrelation]:
        hypers = hyperrelation_extract(tokens,rels) if self.lang=="en" else []
        for f in frames:
            nodes=[f.predicate_token]+[r.filler for r in f.roles]
            rolemap={"predicate":f.predicate_token}
            for r in f.roles:
                rolemap[r.role.lower()]=r.filler
            hypers.append(Hyperrelation(
                id=uid("h"),relation=f.frame_type,nodes=list(dict.fromkeys(nodes)),
                roles=rolemap,confidence=f.confidence,status=f.status
            ))
        return hypers

    def native_transformations(self,text,tokens,typed) -> list[Transformation]:
        out=infer_transformations(typed)
        for op in self.spec.get("operators",[]):
            if self.operator_observed(op,text,tokens):
                out.append(Transformation(
                    id=uid("tr"),source="surface",target=f"{op}_analysis",
                    operation=f"NATIVE::{self.lang}::{op}",
                    affected_spaces=["MORPH","SYN","REF","SCOPE"],
                    delta={},status="OBSERVED_OPERATOR"
                ))
        return out

    def build_map_state(self,name: str | None=None) -> LanguageMapState:
        name=name or self.name
        axes={s:dict(v) for s,v in self.spec.get("axes",{}).items()}
        typed={s:{} for s in SPACE_ORDER}
        for s,vals in axes.items():
            typed[s]={k:float(v) for k,v in vals.items()}

        # Active map relation graph.
        rels=[
            Relation("ORTH","MORPH","CONDITIONS_REALIZATION","MAP",.72,evidence=self.lang),
            Relation("MORPH","SYN","CONSTRAINS","MAP",.86,evidence=self.lang),
            Relation("SYN","ROLE","REALIZES_ROLES","MAP",.82,evidence=self.lang),
            Relation("REF","ROLE","BINDS_ARGUMENTS","MAP",.84,evidence=self.lang),
            Relation("SCOPE","LOG","REALIZES_OPERATORS","MAP",.80,evidence=self.lang),
            Relation("SYN","SEM","COMPOSES","MAP",.84,evidence=self.lang),
            Relation("SEM","PRAG","CONTEXTUALIZES","MAP",.70,evidence=self.lang),
            Relation("INFO","DISC","ORGANIZES","MAP",.76,evidence=self.lang),
        ]
        mechanics=self.spec.get("mechanics",{})
        if mechanics.get("evidentiality"):
            rels.append(Relation("MORPH","EPI","MAY_REALIZE","MAP",.64,evidence=str(mechanics.get("evidentiality"))))
        if mechanics.get("topic_prominence"):
            rels.append(Relation("INFO","SYN","REORDERS","MAP",.70,evidence=str(mechanics.get("topic_prominence"))))

        hypers=[
            Hyperrelation(
                id=uid("h"),relation="NATIVE_FORM_MEANING_INTERFACE",
                nodes=["ORTH","MORPH","SYN","ROLE","REF","SEM"],
                roles={"form":"ORTH/MORPH","structure":"SYN/ROLE/REF","meaning":"SEM"},
                confidence=.80,status="PROVISIONAL_NATIVE_MAP"
            ),
            Hyperrelation(
                id=uid("h"),relation="CONTEXT_FIBER_COUPLING",
                nodes=["PRAG","INFO","DISC","SOC"],
                roles={"context":"PRAG","information":"INFO","discourse":"DISC","social":"SOC"},
                confidence=.68,status="PROVISIONAL_NATIVE_MAP"
            )
        ]

        coverage={s:(1.0 if typed.get(s) else 0.0) for s in SPACE_ORDER}
        # Uncertainty over only map spaces that have native observations.
        observed_typed={s:v for s,v in typed.items() if v}
        unc=make_uncertainty(observed_typed if observed_typed else {"UNC":{"map_unknown":.90}})
        val=make_validation(observed_typed if observed_typed else {"UNC":{"map_unknown":.90}},unc)

        mechanics_meta={
            **mechanics,
            "active_interface":[
                "tokenize","analyze_morphology","analyze_syntax",
                "resolve_reference","resolve_scope","assign_roles",
                "build_meaning_frames","encode_native_spaces","validate"
            ],
        }

        script=self.spec.get("script","Unknown")
        return LanguageMapState(
            id=uid("L"),language=self.lang,language_name=name,
            family=LANGUAGE_FAMILIES.get(self.lang,"Unknown"),
            dialectSet=["General","Regional/Historical variation"],
            orthography={"script":script,"native_axes":typed.get("ORTH",{})},
            phonology={"native_axes":typed.get("PHON",{}),"status":"observed where registry specifies; otherwise UNKNOWN"},
            lexicon={"sense_structure":"typed","resource":"WordNet/OMW + native registry where available"},
            morphology={"type":mechanics.get("morphology","unknown"),"native_axes":typed.get("MORPH",{})},
            syntax={"word_order":mechanics.get("word_order","unknown"),"adposition":mechanics.get("adposition","unknown"),"native_axes":typed.get("SYN",{})},
            semantics={"composition":"through native syntax/role/reference map","status":"typed"},
            logic={"scope":"native operator interface","negation":"language-specific"},
            epistemics={"evidentiality":mechanics.get("evidentiality","unknown"),"native_axes":typed.get("EPI",{})},
            pragmatics={"native_axes":typed.get("PRAG",{}),"context_sensitive":True},
            informationStructure={"native_axes":typed.get("INFO",{}),"topic_prominence":mechanics.get("topic_prominence","unknown")},
            discourse={"cohesion":"typed","reference":"native REF + discourse fibers"},
            sociolinguistics={"register":"context fiber","dialect":"context fiber"},
            contextFibers=["register","domain","speaker","listener","time","task","culture","genre","world_state"],
            typed_spaces=typed,relations=rels,hyperrelations=hypers,
            transformations=[f"NATIVE::{self.lang}::{x}" for x in self.spec.get("operators",[])],
            neighborhoods={"graph":"native space/operator/relation graph","status":"explicit_registry+corpus_extensions"},
            density={"type":"requires corpus observations","status":"UNKNOWN_UNTIL_MEASURED"},
            topology={"type":"typed graph / stratified candidate","status":"PROVISIONAL","localDimension":"data-dependent"},
            uncertainty=unc,
            provenance=Provenance(source=f"ATLAS_NATIVE_LANGUAGE_MAP_V10::{self.lang}",method="native-registry+rule+distributional"),
            validation=val,
            nativeMechanics=mechanics_meta,
            nativeOperators=[{"id":x,"status":"REGISTERED_NATIVE_OPERATOR"} for x in self.spec.get("operators",[])],
            nativeConstraints=[{"id":x,"status":"REGISTERED_NATIVE_CONSTRAINT"} for x in self.spec.get("constraints",[])],
            nativeMetrics={
                "ORTH":"masked feature distance",
                "MORPH":"typed feature + operator distance",
                "SYN":"relation/role graph distance",
                "REF":"coreference graph distance",
                "SEM":"frame/role structural distance",
                "EPI":"evidence/commitment/evidentiality distance",
            },
            alignmentInterfaces=[
                {"source":"ROLE","target":"UNIVERSAL_ROLE_GRAPH","map":"AGENT/EXPERIENCER/THEME/TARGET","information_loss":"tracked"},
                {"source":"REF","target":"DISCOURSE_REFERENT_GRAPH","map":"SPEAKER/ADDRESSEE/SELF/OTHER","information_loss":"tracked"},
                {"source":"SYN","target":"CANONICAL_PREDICATE_ARGUMENT","map":"surface order → role graph","information_loss":"tracked"},
                {"source":"EPI","target":"EPISTEMIC_LATTICE","map":"native evidential/modal realization → typed epistemic state","information_loss":"tracked"},
            ],
            analyzerStatus={
                "class":self.__class__.__name__,
                "version":self.analyzer_version,
                "active":True,
                "native_specificity":"HIGH" if self.lang in {"en","fa","ar","zh","tr"} else "STRUCTURAL_PROFILE",
            },
            observationCoverage=coverage,
        )


class EnglishLanguageMap(AtlasLanguageMap):
    pass


class PersianLanguageMap(AtlasLanguageMap):
    def analyze_pos(self,surface):
        w=surface.casefold()
        if len(surface)==1 and unicodedata.category(surface).startswith("P"): return "PUNCT"
        return NATIVE_LEXICAL_POS["fa"].get(w, "PART" if w in {"را"} else ("ADP" if w in {"از","به","در","با"} else "X"))

    def analyze_morphology(self,surface,pos):
        w=surface.casefold()
        feats=[]
        if w.startswith("می") or w.startswith("نمی"): feats.append("Aspect=Imperfective")
        if w.startswith("ن") and pos=="VERB": feats.append("Polarity=Neg")
        if w.endswith("م"): feats.append("Person=1|Number=Sing")
        if w.endswith("ی"): feats.append("Person=2|Number=Sing")
        if w in {"را"}: feats.append("Case/DOM=RA")
        if w.startswith("خود"): feats.append("Reflexive=Yes")
        return "|".join(feats) if feats else "_"

    def resolve_reference(self,tokens):
        refs=[]; coref=[]
        speaker_tok=next((t for t in tokens if t.lemma=="من"),None)
        add_tok=next((t for t in tokens if t.lemma in {"تو","شما"}),None)
        if speaker_tok:
            refs.append(ReferenceEntity("ref_speaker","SPEAKER","1","singular","SPEAKER",[speaker_tok.id],.97))
        if add_tok:
            refs.append(ReferenceEntity("ref_addressee","ADDRESSEE","2","unknown","ADDRESSEE",[add_tok.id],.94))
        for t in tokens:
            if t.lemma.startswith("خود"):
                if not any(r.id=="ref_speaker" for r in refs):
                    refs.append(ReferenceEntity("ref_speaker","SPEAKER","1","singular","SPEAKER",[],.82))
                ant=speaker_tok.id if speaker_tok else "ref_speaker"
                coref.append(CoreferenceEdge(t.id,ant,"REFLEXIVE_COREFERENCE",.94,"Persian xod- reflexive + person marking"))
        return refs,coref

    def assign_roles_and_frames(self,tokens,coref):
        text=" ".join(t.surface for t in tokens)
        roles=[]; frames=[]
        speaker=next((t for t in tokens if t.lemma=="من"),None)
        self_t=next((t for t in tokens if t.lemma.startswith("خود")),None)
        you=next((t for t in tokens if t.lemma in {"تو","شما"}),None)
        pred=None; ptype=None; val=None
        if "دوست" in text and any("دار" in t.lemma for t in tokens):
            pred=next((t for t in tokens if "دار" in t.lemma),None); ptype="LOVE"; val=.90
        elif any("متنفر" in t.lemma for t in tokens):
            pred=next(t for t in tokens if "متنفر" in t.lemma); ptype="HATE"; val=.10
        if ptype and pred:
            exp_id=speaker.id if speaker else "ref_speaker"
            target=self_t or you
            roles.append(SemanticRoleBinding(pred.id,"EXPERIENCER",exp_id,.93,"Persian native affect frame"))
            if target:
                roles.append(SemanticRoleBinding(pred.id,"TARGET",target.id,.93,"Persian native affect frame"))
            direction="SELF_DIRECTED" if self_t else ("OTHER_DIRECTED" if you else "UNRESOLVED")
            target_label="SELF(SPEAKER)" if self_t else ("ADDRESSEE" if you else "UNKNOWN")
            frames.append(MeaningFrame(
                uid("frame"),"AFFECTIVE_ATTITUDE",ptype,pred.id,[r for r in roles if r.predicate==pred.id],
                "AFFIRMATIVE","ASSERTED",ptype,val,direction,f"{ptype}(SPEAKER, {target_label})",.94,"PROVISIONAL_NATIVE"
            ))
            return roles,frames
        # Epistemic light-verb pattern: باور دارم / فکر می‌کنم
        if "باور" in text and any("دار" in t.lemma for t in tokens):
            pred=next(t for t in tokens if "دار" in t.lemma)
            exp_id=speaker.id if speaker else "ref_speaker"
            roles.append(SemanticRoleBinding(pred.id,"EXPERIENCER",exp_id,.86,"Persian belief light-verb construction"))
            frames.append(MeaningFrame(uid("frame"),"EPISTEMIC_ATTITUDE","BELIEVE",pred.id,roles,
                                       "AFFIRMATIVE","ASSERTED","",None,"",
                                       "BELIEVE(SPEAKER, PROPOSITION)",.84,"PROVISIONAL_NATIVE"))
            return roles,frames
        return super().assign_roles_and_frames(tokens,coref)

    def operator_observed(self,operator,text,tokens):
        joined=" ".join(t.lemma for t in tokens)
        checks={
            "RA_OBJECT_MARKER": any(t.lemma=="را" for t in tokens),
            "EZAFE_LINK": False,
            "LIGHT_VERB_COMPOSE": ("دوست" in joined and "دار" in joined) or ("باور" in joined and "دار" in joined),
            "NA_NEGATION": any(t.lemma.startswith("ن") and t.pos=="VERB" for t in tokens),
            "MI_IMPERFECTIVE": any(t.lemma.startswith("می") or t.lemma.startswith("نمی") for t in tokens),
            "PERSON_SUFFIX": any(t.morph and "Person=" in t.morph for t in tokens),
        }
        return checks.get(operator,False)


class ArabicLanguageMap(AtlasLanguageMap):
    def analyze_pos(self,surface):
        w=surface.casefold()
        if len(surface)==1 and unicodedata.category(surface).startswith("P"): return "PUNCT"
        if w.startswith(("أكره","اكره","أحب","احب","أعتقد","اعتقد")): return "VERB"
        return NATIVE_LEXICAL_POS["ar"].get(w, "ADP" if w in {"من","في","إلى","الى","على"} else "X")

    def analyze_morphology(self,surface,pos):
        w=surface.casefold()
        feats=[]
        if pos=="VERB" and w.startswith(("أ","ا")): feats.append("Person=1|Number=Sing")
        if w.endswith("ك") and len(w)>2: feats.append("ObjectClitic=2")
        if w.startswith("ال") and pos=="NOUN": feats.append("Definite=Yes")
        if w.startswith("نفس"): feats.append("Reflexive=Yes")
        return "|".join(feats) if feats else "_"

    def resolve_reference(self,tokens):
        refs=[];coref=[]
        speaker=next((t for t in tokens if t.lemma in {"أنا","انا"}),None)
        if speaker:
            refs.append(ReferenceEntity("ref_speaker","SPEAKER","1","singular","SPEAKER",[speaker.id],.97))
        if any(t.lemma.endswith("ك") and t.pos=="VERB" for t in tokens) or any(t.lemma in {"أنت","انت","أنتَ","أنتِ"} for t in tokens):
            refs.append(ReferenceEntity("ref_addressee","ADDRESSEE","2","singular","ADDRESSEE",
                                        [t.id for t in tokens if t.lemma in {"أنت","انت","أنتَ","أنتِ"}],.92))
        for t in tokens:
            if t.lemma.startswith("نفس"):
                ant=speaker.id if speaker else "ref_speaker"
                if not speaker:
                    refs.append(ReferenceEntity("ref_speaker","SPEAKER","1","singular","SPEAKER",[],.80))
                coref.append(CoreferenceEdge(t.id,ant,"REFLEXIVE_COREFERENCE",.94,"Arabic nafs + possessive reflexive"))
        return refs,coref

    def assign_roles_and_frames(self,tokens,coref):
        roles=[];frames=[]
        speaker=next((t for t in tokens if t.lemma in {"أنا","انا"}),None)
        self_t=next((t for t in tokens if t.lemma.startswith("نفس")),None)
        love=next((t for t in tokens if t.lemma.startswith(("أحب","احب"))),None)
        hate=next((t for t in tokens if t.lemma.startswith(("أكره","اكره"))),None)
        pred=love or hate
        if pred:
            ptype="LOVE" if love else "HATE"; val=.90 if love else .10
            exp_id=speaker.id if speaker else "ref_speaker"
            roles.append(SemanticRoleBinding(pred.id,"EXPERIENCER",exp_id,.92,"Arabic native affect frame"))
            direction="UNRESOLVED"; target_label="UNKNOWN"
            if self_t:
                roles.append(SemanticRoleBinding(pred.id,"TARGET",self_t.id,.94,"Arabic reflexive target"))
                direction="SELF_DIRECTED";target_label="SELF(SPEAKER)"
            elif pred.lemma.endswith("ك"):
                roles.append(SemanticRoleBinding(pred.id,"TARGET","ref_addressee",.92,"Arabic attached 2nd-person object clitic"))
                direction="OTHER_DIRECTED";target_label="ADDRESSEE"
            frames.append(MeaningFrame(uid("frame"),"AFFECTIVE_ATTITUDE",ptype,pred.id,
                                       [r for r in roles if r.predicate==pred.id],
                                       "AFFIRMATIVE","ASSERTED",ptype,val,direction,
                                       f"{ptype}(SPEAKER, {target_label})",.93,"PROVISIONAL_NATIVE"))
            return roles,frames
        belief=next((t for t in tokens if t.lemma.startswith(("أعتقد","اعتقد"))),None)
        if belief:
            exp_id=speaker.id if speaker else "ref_speaker"
            roles.append(SemanticRoleBinding(belief.id,"EXPERIENCER",exp_id,.88,"Arabic epistemic predicate"))
            frames.append(MeaningFrame(uid("frame"),"EPISTEMIC_ATTITUDE","BELIEVE",belief.id,roles,
                                       "AFFIRMATIVE","ASSERTED","",None,"","BELIEVE(SPEAKER, PROPOSITION)",.86,"PROVISIONAL_NATIVE"))
            return roles,frames
        return super().assign_roles_and_frames(tokens,coref)

    def operator_observed(self,operator,text,tokens):
        return {
            "ATTACHED_PRONOUN":any(t.morph and "ObjectClitic=" in t.morph for t in tokens),
            "AGREEMENT_INFLECT":any(t.morph and "Person=" in t.morph for t in tokens),
            "NEG_PARTICLE":any(t.lemma in {"لا","ليس","لم","لن"} for t in tokens),
            "ROOT_PATTERN_DERIVE":any(t.pos=="VERB" for t in tokens),
            "VSO_SVO_ALTERNATION":False,
        }.get(operator,False)


class MandarinLanguageMap(AtlasLanguageMap):
    def tokenizer_name(self): return "maximal_match_native_zh"

    def tokenize(self,text,utt_id):
        out=[];i=0
        while i<len(text):
            ch=text[i]
            if ch.isspace():
                i+=1;continue
            if unicodedata.category(ch).startswith("P"):
                out.append(_base_token_from_surface(ch,i,i+1,len(out),self.lang,utt_id,pos="PUNCT",morph="_"))
                i+=1;continue
            match=None
            for item in ZH_SEGMENT_LEXICON:
                if text.startswith(item,i):
                    match=item;break
            if match is None:
                match=ch
            pos=self.analyze_pos(match)
            out.append(_base_token_from_surface(match,i,i+len(match),len(out),self.lang,utt_id,pos=pos,morph=self.analyze_morphology(match,pos)))
            i+=len(match)
        return out

    def analyze_pos(self,surface):
        if len(surface)==1 and unicodedata.category(surface).startswith("P"): return "PUNCT"
        return NATIVE_LEXICAL_POS["zh"].get(surface, "X")

    def analyze_morphology(self,surface,pos):
        if surface in {"了"}: return "Aspect=Perfective"
        if surface in {"过","過"}: return "Aspect=Experiential"
        if surface in {"自己","我自己","你自己"}: return "Reflexive=Yes"
        return "_"

    def resolve_reference(self,tokens):
        refs=[];coref=[]
        speaker=next((t for t in tokens if t.surface=="我"),None)
        add=next((t for t in tokens if t.surface=="你"),None)
        if speaker: refs.append(ReferenceEntity("ref_speaker","SPEAKER","1","singular","SPEAKER",[speaker.id],.97))
        if add: refs.append(ReferenceEntity("ref_addressee","ADDRESSEE","2","singular","ADDRESSEE",[add.id],.95))
        for t in tokens:
            if t.surface in {"我自己","自己"} and speaker:
                coref.append(CoreferenceEdge(t.id,speaker.id,"REFLEXIVE_COREFERENCE",.92,"Mandarin ziji / wo-ziji reflexive"))
        return refs,coref

    def assign_roles_and_frames(self,tokens,coref):
        roles=[];frames=[]
        speaker=next((t for t in tokens if t.surface=="我"),None)
        add=next((t for t in tokens if t.surface=="你"),None)
        self_t=next((t for t in tokens if t.surface in {"我自己","自己"}),None)
        pred=next((t for t in tokens if t.surface in {"爱","愛","恨","喜欢","喜歡"}),None)
        if pred:
            ptype="HATE" if pred.surface=="恨" else ("LIKE" if pred.surface in {"喜欢","喜歡"} else "LOVE")
            val=.10 if ptype=="HATE" else (.72 if ptype=="LIKE" else .90)
            exp=speaker.id if speaker else "ref_speaker"
            roles.append(SemanticRoleBinding(pred.id,"EXPERIENCER",exp,.94,"Mandarin SVO affect frame"))
            target=self_t or add
            if target: roles.append(SemanticRoleBinding(pred.id,"TARGET",target.id,.94,"Mandarin affect target"))
            direction="SELF_DIRECTED" if self_t else ("OTHER_DIRECTED" if add else "UNRESOLVED")
            tl="SELF(SPEAKER)" if self_t else ("ADDRESSEE" if add else "UNKNOWN")
            frames.append(MeaningFrame(uid("frame"),"AFFECTIVE_ATTITUDE",ptype,pred.id,
                                       [r for r in roles if r.predicate==pred.id],
                                       "AFFIRMATIVE","ASSERTED",ptype,val,direction,
                                       f"{ptype}(SPEAKER, {tl})",.94,"PROVISIONAL_NATIVE"))
            return roles,frames
        belief=next((t for t in tokens if t.surface in {"认为","認為","相信"}),None)
        if belief:
            exp=speaker.id if speaker else "ref_speaker"
            roles.append(SemanticRoleBinding(belief.id,"EXPERIENCER",exp,.88,"Mandarin epistemic predicate"))
            frames.append(MeaningFrame(uid("frame"),"EPISTEMIC_ATTITUDE","BELIEVE",belief.id,roles,
                                       "AFFIRMATIVE","ASSERTED","",None,"","BELIEVE(SPEAKER, PROPOSITION)",.86,"PROVISIONAL_NATIVE"))
            return roles,frames
        return super().assign_roles_and_frames(tokens,coref)

    def operator_observed(self,operator,text,tokens):
        surfaces={t.surface for t in tokens}
        return {
            "BU_NEGATION":"不" in surfaces,
            "MEI_NEGATION":bool({"没","沒有"} & surfaces),
            "LE_ASPECT":"了" in surfaces,
            "GUO_EXPERIENTIAL":bool({"过","過"} & surfaces),
            "BA_CONSTRUCTION":"把" in surfaces,
            "BEI_PASSIVE":"被" in surfaces,
            "ZAI_PROGRESSIVE":"在" in surfaces,
        }.get(operator,False)


class TurkishLanguageMap(AtlasLanguageMap):
    def analyze_pos(self,surface):
        w=surface.casefold()
        if len(surface)==1 and unicodedata.category(surface).startswith("P"): return "PUNCT"
        if w.startswith("sev") and any(x in w for x in ("iyor","er","di","ecek")): return "VERB"
        if w.startswith("inan") or w.startswith("düşün") or w.startswith("destekle") or w.startswith("ediyor"): return "VERB"
        if w.startswith("kend"): return "PRON"
        if w.startswith("sen") and w.endswith(("den","dan","i","e","a")): return "PRON"
        return NATIVE_LEXICAL_POS["tr"].get(w,"X")

    def analyze_morphology(self,surface,pos):
        w=surface.casefold(); feats=[]
        if w.startswith("kend"): feats.append("Reflexive=Yes")
        if w.endswith(("imi","ımı","umu","ümü")): feats += ["Poss=1SG","Case=Acc"]
        if w.endswith(("den","dan")): feats.append("Case=Abl")
        if "iyor" in w: feats.append("Aspect=Prog")
        if w.endswith(("um","ım","im","üm","yorum","yorum")): feats.append("Person=1|Number=Sing")
        if "ebil" in w or "abil" in w: feats.append("Mood=Potential")
        if "miş" in w or "mış" in w or "muş" in w or "müş" in w: feats.append("Evidential=Inferential")
        return "|".join(dict.fromkeys(feats)) if feats else "_"

    def resolve_reference(self,tokens):
        refs=[];coref=[]
        speaker=next((t for t in tokens if t.lemma=="ben"),None)
        self_t=next((t for t in tokens if t.lemma.startswith("kend")),None)
        add=next((t for t in tokens if t.lemma=="sen" or t.lemma.startswith("sen")),None)
        implicit_1sg=any(t.morph and "Person=1" in t.morph for t in tokens if t.pos=="VERB")
        if speaker or implicit_1sg:
            refs.append(ReferenceEntity("ref_speaker","SPEAKER","1","singular","SPEAKER",[speaker.id] if speaker else [],.94))
        if add:
            refs.append(ReferenceEntity("ref_addressee","ADDRESSEE","2","singular","ADDRESSEE",[add.id],.94))
        if self_t:
            coref.append(CoreferenceEdge(self_t.id,speaker.id if speaker else "ref_speaker","REFLEXIVE_COREFERENCE",.95,"Turkish kendi + possessive/case"))
        return refs,coref

    def assign_roles_and_frames(self,tokens,coref):
        joined=" ".join(t.lemma for t in tokens)
        roles=[];frames=[]
        self_t=next((t for t in tokens if t.lemma.startswith("kend")),None)
        add=next((t for t in tokens if t.lemma.startswith("sen")),None)
        love=next((t for t in tokens if t.lemma.startswith("sev")),None)
        hate_word=next((t for t in tokens if t.lemma=="nefret"),None)
        hate_aux=next((t for t in tokens if t.lemma.startswith("ediyor")),None)
        pred=love or hate_aux
        ptype="LOVE" if love else ("HATE" if hate_word and hate_aux else None)
        if ptype and pred:
            roles.append(SemanticRoleBinding(pred.id,"EXPERIENCER","ref_speaker",.94,"Turkish 1SG agreement/pro-drop"))
            target=self_t or add
            if target: roles.append(SemanticRoleBinding(pred.id,"TARGET",target.id,.94,"Turkish case-marked affect target"))
            direction="SELF_DIRECTED" if self_t else ("OTHER_DIRECTED" if add else "UNRESOLVED")
            tl="SELF(SPEAKER)" if self_t else ("ADDRESSEE" if add else "UNKNOWN")
            val=.90 if ptype=="LOVE" else .10
            frames.append(MeaningFrame(uid("frame"),"AFFECTIVE_ATTITUDE",ptype,pred.id,
                                       [r for r in roles if r.predicate==pred.id],
                                       "AFFIRMATIVE","ASSERTED",ptype,val,direction,
                                       f"{ptype}(SPEAKER, {tl})",.94,"PROVISIONAL_NATIVE"))
            return roles,frames
        belief=next((t for t in tokens if t.lemma.startswith("inan")),None)
        if belief:
            roles.append(SemanticRoleBinding(belief.id,"EXPERIENCER","ref_speaker",.92,"Turkish 1SG agreement"))
            frames.append(MeaningFrame(uid("frame"),"EPISTEMIC_ATTITUDE","BELIEVE",belief.id,roles,
                                       "AFFIRMATIVE","ASSERTED","",None,"","BELIEVE(SPEAKER, PROPOSITION)",.88,"PROVISIONAL_NATIVE"))
            return roles,frames
        return super().assign_roles_and_frames(tokens,coref)

    def operator_observed(self,operator,text,tokens):
        morph="|".join(t.morph for t in tokens)
        return {
            "VOWEL_HARMONY":any(t.pos!="PUNCT" for t in tokens),
            "CASE_SUFFIX":"Case=" in morph,
            "PERSON_SUFFIX":"Person=" in morph,
            "NEGATIVE_SUFFIX":any(re.search(r"m[aeıiuüöo]",t.lemma) for t in tokens if t.pos=="VERB"),
            "EVIDENTIAL_MIS":"Evidential=Inferential" in morph,
            "PROGRESSIVE_IYOR":"Aspect=Prog" in morph,
        }.get(operator,False)


ACTIVE_MAP_CLASSES = {
    "en":EnglishLanguageMap,
    "fa":PersianLanguageMap,
    "ar":ArabicLanguageMap,
    "zh":MandarinLanguageMap,
    "tr":TurkishLanguageMap,
}

@st.cache_resource(show_spinner=False)
def get_active_language_map(lang: str) -> AtlasLanguageMap:
    cls=ACTIVE_MAP_CLASSES.get(lang,AtlasLanguageMap)
    return cls(lang)

def language_map_profile(lang: str, name: str) -> LanguageMapState:
    return get_active_language_map(lang).build_map_state(name)

def native_map_space_summary(lm: LanguageMapState) -> dict[str,float]:
    return {
        s:(mean(lm.typed_spaces.get(s,{}).values()) if lm.typed_spaces.get(s) else float("nan"))
        for s in SPACE_ORDER
    }

def compare_language_maps_native(a: LanguageMapState,b: LanguageMapState) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Compare shared coordinates only. Missing-native-axis ≠ zero.
    Returns:
      space summary dataframe
      coordinate alignment dataframe
    """
    coord_rows=[]
    space_rows=[]
    for sp in SPACE_ORDER:
        ca=a.typed_spaces.get(sp,{})
        cb=b.typed_spaces.get(sp,{})
        shared=sorted(set(ca)&set(cb))
        if shared:
            diffs=[abs(float(ca[k])-float(cb[k])) for k in shared]
            coherence=clamp(1-mean(diffs))
            mean_a=mean(float(ca[k]) for k in shared)
            mean_b=mean(float(cb[k]) for k in shared)
            status="COMPARABLE_SHARED_AXES"
            for k in shared:
                coord_rows.append({
                    "space":sp,"coordinate":k,
                    a.language_name:float(ca[k]),b.language_name:float(cb[k]),
                    "delta":float(cb[k])-float(ca[k]),
                    "status":"SHARED_NATIVE_AXIS"
                })
        else:
            coherence=float("nan");mean_a=float("nan");mean_b=float("nan")
            status="NO_SHARED_NATIVE_AXES"
        space_rows.append({
            "space":sp,
            f"{a.language_name}_shared_axis_mean":mean_a,
            f"{b.language_name}_shared_axis_mean":mean_b,
            "shared_coordinate_count":len(shared),
            "native_axis_coherence":coherence,
            "status":status
        })
    return pd.DataFrame(space_rows),pd.DataFrame(coord_rows)

def native_language_map_graph(lm: LanguageMapState) -> go.Figure:
    G=nx.MultiDiGraph()
    for sp,coords in lm.typed_spaces.items():
        if coords:
            G.add_node(sp,label=sp,kind="SPACE")
            for c,v in coords.items():
                node=f"{sp}.{c}"
                G.add_node(node,label=c,kind="COORDINATE",value=v)
                G.add_edge(sp,node,relation="HAS_AXIS")
    for r in lm.relations:
        G.add_node(r.source,label=r.source,kind="SPACE")
        G.add_node(r.target,label=r.target,kind="SPACE")
        G.add_edge(r.source,r.target,relation=r.relation)

    if not G.nodes:
        return go.Figure()

    pos=nx.spring_layout(G,seed=42,k=1.3)
    ex=[];ey=[]
    for u,v in G.edges():
        x0,y0=pos[u];x1,y1=pos[v]
        ex += [x0,x1,None];ey += [y0,y1,None]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1),hoverinfo="skip"))
    xs=[];ys=[];labels=[];hover=[]
    for n,d in G.nodes(data=True):
        x,y=pos[n];xs.append(x);ys.append(y);labels.append(d.get("label",n))
        hover.append(f"<b>{d.get('label',n)}</b><br>{d.get('kind','')}<br>value={d.get('value','')}")
    fig.add_trace(go.Scatter(
        x=xs,y=ys,mode="markers+text",text=labels,textposition="top center",
        customdata=hover,hovertemplate="%{customdata}<extra></extra>",marker=dict(size=8)
    ))
    fig.update_layout(template="plotly_dark",height=720,showlegend=False,
                      title=f"Active native language map — {lm.language_name}",
                      xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig


# =============================================================================
# Token / utterance / language-map constructors
# =============================================================================

def make_token_states(text: str, lang: str, utt_id: str) -> list[TokenState]:
    analyzer=get_active_language_map(lang)
    tokens=analyzer.tokenize(text,utt_id)
    spaces=[analyzer.encode_native_token_state(t,text) for t in tokens]

    # Provisional roles/coreference are needed so token-local relation views
    # already reflect the active map.
    refs,coref=analyzer.resolve_reference(tokens)
    roles,frames=analyzer.assign_roles_and_frames(tokens,coref)
    rels=analyzer.extract_relations(tokens,spaces,roles,coref)
    hypers=analyzer.extract_hyperrelations(tokens,rels,frames)

    out=[]
    for tok,sp in zip(tokens,spaces):
        local_rels=[r for r in rels if r.source==tok.id or r.target==tok.id]
        local_h=[h for h in hypers if tok.id in h.nodes]
        unc=make_uncertainty(sp)
        val=make_validation(sp,unc)
        out.append(TokenState(
            token=tok,typed_spaces=sp,relations=local_rels,hyperrelations=local_h,
            uncertainty=unc,
            provenance=Provenance(
                source=f"ATLAS_NATIVE_TOKEN_ENCODER_V10::{lang}",
                method=f"{analyzer.__class__.__name__}+typed-state"
            ),
            validation=val,
            lexicalNeighborhood=build_lexical_neighborhood(tok),
        ))
    return out

def make_utterance_state(text: str, lang: str) -> UtteranceState:
    utt_id=uid("u")
    analyzer=get_active_language_map(lang)

    token_states=make_token_states(text,lang,utt_id)
    tokens=[ts.token for ts in token_states]
    spaces=[ts.typed_spaces for ts in token_states]

    refs,coref=analyzer.resolve_reference(tokens)
    semantic_roles,meaning_frames=analyzer.assign_roles_and_frames(tokens,coref)

    typed=aggregate_spaces(token_states)
    rels=analyzer.extract_relations(tokens,spaces,semantic_roles,coref)
    hypers=analyzer.extract_hyperrelations(tokens,rels,meaning_frames)

    unc=make_uncertainty(typed)
    val=make_validation(typed,unc)
    ints=candidate_interpretations(text,typed)
    transformations=analyzer.native_transformations(text,tokens,typed)

    certificate=build_comprehension_certificate(tokens,semantic_roles,meaning_frames,coref)
    attention=build_structural_attention(token_states,semantic_roles,meaning_frames)
    attention["uncertainty_focus"]=unc.vector
    attention["native_language_map"]=analyzer.__class__.__name__

    comprehension={
        "preferred_interpretation":ints[0]["id"],
        "candidate_count":len(ints),
        "mean_token_integration":mean(ts.typed_spaces["COMP"]["integration"] for ts in token_states),
        "unresolved_scope":typed["SCOPE"]["scope_ambiguity"],
        "certificate_confidence":certificate.confidence,
        "resolved_items":{
            "predicate":certificate.predicate_resolved,
            "subject":certificate.subject_resolved,
            "object":certificate.object_resolved,
            "coreference":certificate.coreference_resolved,
            "affect_target":certificate.affect_target_resolved,
        },
        "native_analyzer":analyzer.__class__.__name__,
        "status":"PROVISIONAL",
    }

    knowledge_delta={
        "update_potential":typed["KNOW"]["updateability"],
        "grounding":typed["KNOW"]["grounding"],
        "factivity":typed["KNOW"]["factivity"],
        "status":"CANDIDATE_DELTA",
    }

    state_delta=build_state_delta(typed,meaning_frames)
    neighbors=generate_counterfactual_neighbors(text,meaning_frames) if lang=="en" else []

    return UtteranceState(
        id=utt_id,form=text,language=lang,tokens=token_states,
        typed_spaces=typed,relations=rels,hyperrelations=hypers,
        transformations=transformations,candidateInterpretations=ints,
        uncertainty=unc,attention=attention,comprehension=comprehension,
        knowledgeDelta=knowledge_delta,
        provenance=Provenance(
            source=f"ATLAS_ACTIVE_LANGUAGE_PIPELINE_V10::{lang}",
            method=f"{analyzer.__class__.__name__}:native-map→typed-state"
        ),
        validation=val,references=refs,coreference=coref,
        semanticRoles=semantic_roles,meaningFrames=meaning_frames,
        comprehensionCertificate=certificate,stateDelta=state_delta,
        counterfactualNeighbors=neighbors,
        languageMapAnalyzer=analyzer.__class__.__name__,
        nativeMechanicsObserved=analyzer.native_mechanics_observed(text,tokens),
    )


# =============================================================================
# Embedding layer
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_sentence_transformer(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

def hashed_embeddings(texts: list[str], dims: int) -> np.ndarray:
    half = max(16, dims // 2)
    word = HashingVectorizer(
        n_features=half, ngram_range=(1,2), analyzer="word",
        alternate_sign=False, norm="l2"
    )
    char = HashingVectorizer(
        n_features=max(16, dims-half), ngram_range=(2,5), analyzer="char_wb",
        alternate_sign=False, norm="l2"
    )
    x = np.hstack([word.transform(texts).toarray(), char.transform(texts).toarray()])
    return normalize_rows(x)

def embed(texts: list[str], model_name: str, offline: bool, dims: int) -> tuple[np.ndarray,str]:
    if not offline:
        try:
            model = load_sentence_transformer(model_name)
            x = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
            return normalize_rows(x), model_name
        except Exception as exc:
            st.warning(f"SentenceTransformer unavailable; falling back to hashing: {exc}")
    return hashed_embeddings(texts, dims), f"offline-hashed-{dims}d"

def train_autoencoder(x: np.ndarray, latent_dim: int, hidden_dim: int, epochs: int, lr: float, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x = np.asarray(x, dtype=np.float32)
    in_dim = x.shape[1]
    latent_dim = max(2, min(latent_dim, in_dim))
    hidden_dim = max(latent_dim * 2, min(hidden_dim, in_dim * 2))

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
            self.dec = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim))
        def forward(self, a):
            z = self.enc(a)
            return self.dec(z), z

    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tx = torch.from_numpy(x)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        recon, _ = model(tx)
        loss = loss_fn(recon, tx)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        recon, z = model(tx)
    return z.numpy(), recon.numpy(), losses



# =============================================================================
# Batch language benchmark / intelligence report engine
# =============================================================================

def parse_language_benchmark_input(raw: str) -> list[tuple[str, str, str]]:
    """
    Accept either:

    [English]
    I love myself.
    I hate you.

    [Farsi / Persian]
    ...

    or:
    English | I love myself.
    English | I hate you.

    Returns (language_name, language_code, utterance).
    """
    rows: list[tuple[str,str,str]] = []
    current_language: str | None = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # [Language Name]
        if line.startswith("[") and line.endswith("]"):
            candidate = line[1:-1].strip()
            if candidate in LANGUAGES:
                current_language = candidate
            else:
                current_language = None
            continue

        # Language | utterance
        if "|" in line:
            maybe_lang, utt = [x.strip() for x in line.split("|", 1)]
            if maybe_lang in LANGUAGES and utt:
                rows.append((maybe_lang, LANGUAGES[maybe_lang], utt))
                continue

        if current_language and line:
            rows.append((current_language, LANGUAGES[current_language], line))

    return rows


def _content_tokens(state: UtteranceState) -> list[TokenState]:
    return [ts for ts in state.tokens if ts.token.pos != "PUNCT"]


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _lexical_coverage_for_states(states: list[UtteranceState]) -> tuple[float,float,dict[str,int]]:
    content = [ts for s in states for ts in _content_tokens(s)]
    if not content:
        return 0.0, 0.0, {}

    with_lex = [
        ts for ts in content
        if ts.lexicalNeighborhood
        and (
            ts.lexicalNeighborhood.senses
            or ts.lexicalNeighborhood.edges
        )
    ]
    edge_count = sum(
        len(ts.lexicalNeighborhood.edges)
        for ts in content if ts.lexicalNeighborhood
    )

    inv: dict[str,int] = {}
    for ts in content:
        ln = ts.lexicalNeighborhood
        if not ln:
            continue
        for e in ln.edges:
            inv[e.relation] = inv.get(e.relation,0) + 1

    return (
        _safe_ratio(len(with_lex), len(content)),
        _safe_ratio(edge_count, len(content)),
        inv
    )


def benchmark_failure_findings(language_name: str, state: UtteranceState) -> list[BenchmarkFinding]:
    out: list[BenchmarkFinding] = []
    content = _content_tokens(state)

    analyzer_name = state.languageMapAnalyzer or ""
    if analyzer_name == "AtlasLanguageMap" and state.language != "und":
        out.append(BenchmarkFinding(
            category="LANGUAGE_MAP",
            label="STRUCTURAL_PROFILE_ONLY",
            severity="MEDIUM",
            language=language_name,
            utterance=state.form,
            evidence=f"{state.language} is using the conservative structural-profile analyzer rather than a dedicated native subclass",
            recommendation="Implement a dedicated active native analyzer for this language before treating failures as linguistic findings."
        ))

    # POS / lexical failures
    for ts in content:
        if ts.token.pos == "X":
            out.append(BenchmarkFinding(
                category="LEXICAL_ANALYSIS",
                label="UNRESOLVED_POS",
                severity="HIGH",
                language=language_name,
                utterance=state.form,
                evidence=f"token `{ts.token.surface}` classified as X",
                recommendation="Add language-specific lexical/POS evidence or parser support."
            ))

        ln = ts.lexicalNeighborhood
        if ln and ln.status in {"RESOURCE_UNAVAILABLE","NO_DICTIONARY_ENTRY"} and ts.token.pos not in {"PRON","DET","AUX","PART","SCONJ","FUNC"}:
            out.append(BenchmarkFinding(
                category="LEXICAL_RESOURCE",
                label="LEXICAL_NEIGHBORHOOD_MISSING",
                severity="MEDIUM",
                language=language_name,
                utterance=state.form,
                evidence=f"`{ts.token.surface}` has lexical status {ln.status}",
                recommendation="Add WordNet/OMW/native lexicon or curated language-specific lexical registry."
            ))

    cert = state.comprehensionCertificate
    if cert:
        if not cert.predicate_resolved:
            out.append(BenchmarkFinding(
                category="STRUCTURE",
                label="PREDICATE_UNRESOLVED",
                severity="HIGH",
                language=language_name,
                utterance=state.form,
                evidence="comprehension certificate predicate_resolved=False",
                recommendation="Improve predicate identification / language-specific parsing."
            ))
        if not cert.subject_resolved:
            out.append(BenchmarkFinding(
                category="STRUCTURE",
                label="SUBJECT_OR_EXPERIENCER_UNRESOLVED",
                severity="MEDIUM",
                language=language_name,
                utterance=state.form,
                evidence="subject/experiencer not resolved",
                recommendation="Add semantic-role/dependency parsing for this construction."
            ))
        if not cert.object_resolved:
            out.append(BenchmarkFinding(
                category="STRUCTURE",
                label="OBJECT_OR_TARGET_UNRESOLVED",
                severity="MEDIUM",
                language=language_name,
                utterance=state.form,
                evidence="object/target not resolved",
                recommendation="Improve argument-structure and semantic-role mapping."
            ))
        if not cert.coreference_resolved:
            out.append(BenchmarkFinding(
                category="REFERENCE",
                label="COREFERENCE_UNRESOLVED",
                severity="HIGH",
                language=language_name,
                utterance=state.form,
                evidence="coreference_resolved=False",
                recommendation="Add language-specific reference/reflexive/coreference rules."
            ))

    if state.validation.confidence < .55:
        out.append(BenchmarkFinding(
            category="VALIDATION",
            label="LOW_VALIDATION_CONFIDENCE",
            severity="MEDIUM",
            language=language_name,
            utterance=state.form,
            evidence=f"validation confidence={state.validation.confidence:.3f}",
            recommendation="Increase observed coordinate coverage and reduce shared/default heuristics."
        ))

    if mean(state.uncertainty.vector.values(),0.0) > .55:
        out.append(BenchmarkFinding(
            category="UNCERTAINTY",
            label="HIGH_UNCERTAINTY",
            severity="MEDIUM",
            language=language_name,
            utterance=state.form,
            evidence=f"mean uncertainty={mean(state.uncertainty.vector.values(),0.0):.3f}",
            recommendation="Acquire stronger language-specific observations and structural evidence."
        ))

    # Frame gap: content utterance with no frame
    content_predicates = [ts for ts in content if ts.token.pos in {"VERB","ADJ","AUX"}]
    if content_predicates and not state.meaningFrames:
        out.append(BenchmarkFinding(
            category="MEANING_FRAME",
            label="FRAME_NOT_CONSTRUCTED",
            severity="HIGH",
            language=language_name,
            utterance=state.form,
            evidence="predicate-like material exists but no meaning frame was emitted",
            recommendation="Add/extend frame registry for this predicate/construction."
        ))

    return out


def build_language_intelligence_profile(
    language_name: str,
    language_code: str,
    states: list[UtteranceState],
    findings: list[BenchmarkFinding],
) -> LanguageIntelligenceProfile:
    token_count = sum(len(_content_tokens(s)) for s in states)
    lexical_coverage, relation_density, lex_inv = _lexical_coverage_for_states(states)

    frame_count = sum(len(s.meaningFrames) for s in states)
    role_count = sum(len(s.semanticRoles) for s in states)

    frame_coverage = _safe_ratio(
        sum(1 for s in states if s.meaningFrames),
        len(states)
    )
    role_coverage = _safe_ratio(
        sum(1 for s in states if s.semanticRoles),
        len(states)
    )

    ref_relevant = [
        s for s in states
        if any(ts.token.pos == "PRON" for ts in _content_tokens(s))
    ]
    reference_resolution = (
        _safe_ratio(sum(1 for s in ref_relevant if s.references), len(ref_relevant))
        if ref_relevant else 1.0
    )

    reflexive_states = [
        s for s in states
        if any(ts.token.lemma in REFLEXIVE_TO_PERSON for ts in _content_tokens(s))
    ]
    coref_resolution = (
        _safe_ratio(sum(1 for s in reflexive_states if s.coreference), len(reflexive_states))
        if reflexive_states else 1.0
    )

    comp_conf = mean(
        [
            s.comprehensionCertificate.confidence
            for s in states if s.comprehensionCertificate
        ],
        0.0
    )
    val_conf = mean([s.validation.confidence for s in states],0.0)
    unc = mean(
        [mean(s.uncertainty.vector.values(),0.0) for s in states],
        0.0
    )

    typed_cov = {}
    for sp in SPACE_ORDER:
        vals = [space_summary(s.typed_spaces).get(sp,0.0) for s in states]
        # Coverage as activation/availability proxy, not truth.
        typed_cov[sp] = mean(vals,0.0)

    frame_types: dict[str,int] = {}
    structural_relations: dict[str,int] = {}
    for s in states:
        for f in s.meaningFrames:
            frame_types[f.frame_type] = frame_types.get(f.frame_type,0) + 1
        for r in s.relations:
            structural_relations[r.relation] = structural_relations.get(r.relation,0) + 1

    return LanguageIntelligenceProfile(
        language=language_code,
        language_name=language_name,
        utterance_count=len(states),
        token_count=token_count,
        lexical_coverage=lexical_coverage,
        lexical_relation_density=relation_density,
        meaning_frame_coverage=frame_coverage,
        semantic_role_coverage=role_coverage,
        reference_resolution=reference_resolution,
        coreference_resolution=coref_resolution,
        comprehension_confidence=comp_conf,
        validation_confidence=val_conf,
        uncertainty=unc,
        typed_space_coverage=typed_cov,
        frame_type_counts=frame_types,
        relation_type_counts=structural_relations,
        lexical_relation_counts=lex_inv,
        failure_count=sum(1 for f in findings if f.language == language_name),
        native_map_coverage=mean([
            mean([
                1.0 if ts.typed_spaces.get(sp) and any(k.startswith("native::") for k in ts.typed_spaces.get(sp,{})) else 0.0
                for sp in ("ORTH","PHON","MORPH","SYN","REF","SCOPE","EPI","INFO","PRAG","ROLE")
            ])
            for s in states for ts in _content_tokens(s)
        ],0.0),
        native_analyzer=(states[0].languageMapAnalyzer if states else ""),
    )


def benchmark_dimension_scores(profiles: list[LanguageIntelligenceProfile]) -> dict[str,float]:
    if not profiles:
        return {}
    lexical = mean([p.lexical_coverage for p in profiles])
    structural = mean([
        mean([p.meaning_frame_coverage,p.semantic_role_coverage])
        for p in profiles
    ])
    referential = mean([
        mean([p.reference_resolution,p.coreference_resolution])
        for p in profiles
    ])
    comprehension = mean([p.comprehension_confidence for p in profiles])
    validation = mean([p.validation_confidence for p in profiles])
    uncertainty_control = mean([1-p.uncertainty for p in profiles])
    cross_language_balance = 1.0 - float(np.std([
        mean([
            p.lexical_coverage,
            p.meaning_frame_coverage,
            p.semantic_role_coverage,
            p.comprehension_confidence
        ])
        for p in profiles
    ]))

    return {
        "lexical_intelligence": clamp(lexical),
        "structural_intelligence": clamp(structural),
        "referential_intelligence": clamp(referential),
        "comprehension_intelligence": clamp(comprehension),
        "validation_quality": clamp(validation),
        "uncertainty_control": clamp(uncertainty_control),
        "cross_language_balance": clamp(cross_language_balance),
    }


def build_intelligence_benchmark_report(
    grouped_states: dict[str, list[UtteranceState]],
    language_codes: dict[str,str],
    embeddings: np.ndarray | None = None,
    labels: list[str] | None = None,
    discovery: DiscoveryReport | None = None,
) -> IntelligenceBenchmarkReport:
    all_states = [s for states in grouped_states.values() for s in states]
    findings: list[BenchmarkFinding] = []

    for lang_name, states in grouped_states.items():
        for s in states:
            findings.extend(benchmark_failure_findings(lang_name,s))

    profiles = [
        build_language_intelligence_profile(
            lang_name,
            language_codes[lang_name],
            states,
            findings
        )
        for lang_name,states in grouped_states.items()
    ]

    global_space_cov = {}
    for sp in SPACE_ORDER:
        global_space_cov[sp] = mean([
            p.typed_space_coverage.get(sp,0.0) for p in profiles
        ],0.0)

    frame_inv: dict[str,int] = {}
    rel_inv: dict[str,int] = {}
    lex_inv: dict[str,int] = {}

    for p in profiles:
        for k,v in p.frame_type_counts.items():
            frame_inv[k] = frame_inv.get(k,0)+v
        for k,v in p.relation_type_counts.items():
            rel_inv[k] = rel_inv.get(k,0)+v
        for k,v in p.lexical_relation_counts.items():
            lex_inv[k] = lex_inv.get(k,0)+v

    alignment = None
    invariant = None
    residual_count = 0

    if embeddings is not None and labels and len(all_states) >= 2:
        try:
            alignment, invariant, residuals, _ = align_many(
                all_states, embeddings, labels
            )
            residual_count = len(residuals)
        except Exception:
            alignment = None
            invariant = None

    dimensions = benchmark_dimension_scores(profiles)

    global_metrics = {
        "mean_lexical_coverage": mean([p.lexical_coverage for p in profiles],0.0),
        "mean_frame_coverage": mean([p.meaning_frame_coverage for p in profiles],0.0),
        "mean_role_coverage": mean([p.semantic_role_coverage for p in profiles],0.0),
        "mean_reference_resolution": mean([p.reference_resolution for p in profiles],0.0),
        "mean_coreference_resolution": mean([p.coreference_resolution for p in profiles],0.0),
        "mean_comprehension_confidence": mean([p.comprehension_confidence for p in profiles],0.0),
        "mean_validation_confidence": mean([p.validation_confidence for p in profiles],0.0),
        "mean_uncertainty": mean([p.uncertainty for p in profiles],0.0),
        "failure_count": float(len(findings)),
    }

    return IntelligenceBenchmarkReport(
        id=uid("intel"),
        title="ATLAS Language Intelligence Benchmark",
        total_languages=len(grouped_states),
        total_utterances=len(all_states),
        total_tokens=sum(len(_content_tokens(s)) for s in all_states),
        language_profiles=profiles,
        global_metrics=global_metrics,
        typed_space_coverage=global_space_cov,
        frame_inventory=frame_inv,
        structural_relation_inventory=rel_inv,
        lexical_relation_inventory=lex_inv,
        findings=findings,
        cross_language_alignment=alignment,
        invariant_candidate=invariant,
        residual_count=residual_count,
        discovery=discovery,
        benchmark_dimensions=dimensions,
    )


def intelligence_profile_df(report: IntelligenceBenchmarkReport) -> pd.DataFrame:
    return pd.DataFrame([{
        "language": p.language_name,
        "code": p.language,
        "utterances": p.utterance_count,
        "tokens": p.token_count,
        "lexical_coverage": p.lexical_coverage,
        "lexical_relation_density": p.lexical_relation_density,
        "frame_coverage": p.meaning_frame_coverage,
        "role_coverage": p.semantic_role_coverage,
        "reference_resolution": p.reference_resolution,
        "coreference_resolution": p.coreference_resolution,
        "comprehension": p.comprehension_confidence,
        "validation": p.validation_confidence,
        "uncertainty": p.uncertainty,
        "native_map_coverage": p.native_map_coverage,
        "native_analyzer": p.native_analyzer,
        "failures": p.failure_count,
    } for p in report.language_profiles])


def intelligence_findings_df(report: IntelligenceBenchmarkReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in report.findings])


def benchmark_dimension_df(report: IntelligenceBenchmarkReport) -> pd.DataFrame:
    return pd.DataFrame([
        {"dimension":k,"score":v}
        for k,v in report.benchmark_dimensions.items()
    ])


def intelligence_report_markdown(
    report: IntelligenceBenchmarkReport,
    grouped_states: dict[str,list[UtteranceState]],
    embedding_model: str,
) -> str:
    lines = [f"# {report.title}", ""]

    lines += ["## Executive Intelligence Summary", ""]
    lines.append(f"- Languages tested: **{report.total_languages}**")
    lines.append(f"- Utterances tested: **{report.total_utterances}**")
    lines.append(f"- Content tokens tested: **{report.total_tokens}**")
    lines.append(f"- Findings / failures: **{len(report.findings)}**")
    lines.append(f"- Residual observations: **{report.residual_count}**")
    lines.append("")
    lines.append("> Scores are provisional analyzer-performance indicators, not claims of human-like intelligence.")
    lines.append("")

    lines += ["## Benchmark Dimensions", ""]
    for k,v in report.benchmark_dimensions.items():
        lines.append(f"- `{k}`: **{v:.3f}**")
    lines.append("")

    lines += ["## Global Metrics", ""]
    for k,v in report.global_metrics.items():
        lines.append(f"- `{k}`: **{v:.3f}**")
    lines.append("")

    lines += ["## Language Intelligence Profiles", ""]
    for p in report.language_profiles:
        lines.append(f"### {p.language_name} (`{p.language}`)")
        lines.append(f"- Utterances: {p.utterance_count}")
        lines.append(f"- Tokens: {p.token_count}")
        lines.append(f"- Lexical coverage: {p.lexical_coverage:.3f}")
        lines.append(f"- Lexical relation density: {p.lexical_relation_density:.3f}")
        lines.append(f"- Meaning-frame coverage: {p.meaning_frame_coverage:.3f}")
        lines.append(f"- Semantic-role coverage: {p.semantic_role_coverage:.3f}")
        lines.append(f"- Reference resolution: {p.reference_resolution:.3f}")
        lines.append(f"- Coreference resolution: {p.coreference_resolution:.3f}")
        lines.append(f"- Comprehension confidence: {p.comprehension_confidence:.3f}")
        lines.append(f"- Validation confidence: {p.validation_confidence:.3f}")
        lines.append(f"- Mean uncertainty: {p.uncertainty:.3f}")
        lines.append(f"- Active native analyzer: `{p.native_analyzer}`")
        lines.append(f"- Native-map coordinate coverage: {p.native_map_coverage:.3f}")
        lines.append(f"- Failure count: {p.failure_count}")
        lines.append("")

    lines += ["## Meaning-Frame Inventory", ""]
    if report.frame_inventory:
        for k,v in sorted(report.frame_inventory.items(),key=lambda x:x[1],reverse=True):
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- No frames constructed.")
    lines.append("")

    lines += ["## Structural Relation Inventory", ""]
    for k,v in sorted(report.structural_relation_inventory.items(),key=lambda x:x[1],reverse=True):
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines += ["## Lexical Relation Inventory", ""]
    if report.lexical_relation_inventory:
        for k,v in sorted(report.lexical_relation_inventory.items(),key=lambda x:x[1],reverse=True):
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- No lexical relation evidence available.")
    lines.append("")

    lines += ["## Typed-Space Coverage / Activation", ""]
    for sp,v in sorted(report.typed_space_coverage.items(),key=lambda x:x[1],reverse=True):
        lines.append(f"- `{sp}`: {v:.3f}")
    lines.append("")

    lines += ["## Cross-Language Alignment / Invariant Candidates", ""]
    if report.cross_language_alignment:
        lines.append(f"- Alignment status: `{report.cross_language_alignment.status}`")
        lines.append(f"- Embedding coherence: {report.cross_language_alignment.embedding_coherence:.3f}")
    else:
        lines.append("- Cross-language alignment not available.")
    if report.invariant_candidate:
        lines.append(f"- Invariant status: `{report.invariant_candidate.status}`")
        lines.append(f"- Invariant confidence: {report.invariant_candidate.confidence:.3f}")
        lines.append(
            "- Shared typed coordinates/spaces: " +
            (", ".join(report.invariant_candidate.shared_coordinates.keys()) or "none")
        )
    lines.append("")

    lines += ["## Failure / Falsification Ledger", ""]
    if report.findings:
        severity_order = {"HIGH":0,"MEDIUM":1,"LOW":2}
        for f in sorted(report.findings,key=lambda x:(severity_order.get(x.severity,9),x.language,x.category)):
            lines.append(
                f"- **{f.severity} / {f.category} / {f.label}** — "
                f"{f.language}: `{f.utterance}`"
            )
            lines.append(f"  - Evidence: {f.evidence}")
            lines.append(f"  - Repair: {f.recommendation}")
    else:
        lines.append("- No benchmark failures emitted.")
    lines.append("")

    lines += ["## Structural Discovery", ""]
    if report.discovery:
        lines.append(f"- Candidate coordinates: {len(report.discovery.coordinate_candidates)}")
        for c in report.discovery.coordinate_candidates[:20]:
            lines.append(
                f"  - `{c.proposed_name}` novelty={c.novelty_score:.3f}, "
                f"EVR={c.explained_variance:.3f}"
            )
        lines.append(f"- Persistent gaps: {len(report.discovery.gap_candidates)}")
        for g in report.discovery.gap_candidates[:20]:
            lines.append(
                f"  - `{g.location}` persistence={g.persistence:.3f}"
            )
        lines.append(f"- Term proposals: {len(report.discovery.term_proposals)}")
    else:
        lines.append("- Discovery was not run for this benchmark.")
    lines.append("")

    lines += ["## Utterance-Level Intelligence Evidence", ""]
    for lang_name, states in grouped_states.items():
        lines.append(f"### {lang_name}")
        for s in states:
            lines.append(f"#### {s.form}")
            if s.meaningFrames:
                for mf in s.meaningFrames:
                    lines.append(f"- Meaning frame: `{mf.canonical_form}`")
            else:
                lines.append("- Meaning frame: unresolved")
            if s.comprehensionCertificate:
                lines.append(
                    f"- Comprehension certificate: {s.comprehensionCertificate.confidence:.3f}; "
                    f"unresolved={s.comprehensionCertificate.unresolved_items or 'none'}"
                )
            lines.append(
                f"- Relations: {len(s.relations)}; hyperrelations: {len(s.hyperrelations)}; "
                f"counterfactuals: {len(s.counterfactualNeighbors)}"
            )
        lines.append("")

    lines += ["## Provenance / Interpretation Rules", ""]
    lines.append(f"- Embedding model: `{embedding_model}`")
    lines.append("- This report measures ATLAS prototype coverage, structure construction, uncertainty, and cross-language consistency.")
    lines.append("- It does not equate a high score with general intelligence.")
    lines.append("- Unknown/default coordinates do not count as evidence of invariance.")
    lines.append("- Language-specific analyzers remain required for strong multilingual claims.")
    lines.append("- Persistent cross-language residuals may indicate missing coordinates, relations, context fibers, metrics, or lexicalizations.")
    lines.append("")

    return "\n".join(lines)



# =============================================================================
# Structural gap / coordinate discovery support
# =============================================================================

UNKNOWN_MARKERS = (
    "_unknown",
    "unknown_",
    "requires_native_analyzer",
)

def coordinate_is_observed(space: str, coordinate: str, value: float) -> bool:
    key = coordinate.casefold()
    if any(mark in key for mark in UNKNOWN_MARKERS):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False

def flatten_observed_coordinates(state: UtteranceState) -> dict[str, float]:
    flat: dict[str, float] = {}
    for space, coords in state.typed_spaces.items():
        for coordinate, value in coords.items():
            try:
                value = float(value)
            except Exception:
                continue
            if coordinate_is_observed(space, coordinate, value):
                flat[f"{space}.{coordinate}"] = value
    return flat

def build_observed_feature_matrix(
    states: list[UtteranceState],
) -> tuple[np.ndarray, list[str], list[str]]:
    flattened = [flatten_observed_coordinates(s) for s in states]
    names = sorted({k for d in flattened for k in d})

    if not names:
        return np.empty((len(states), 0)), [], []

    raw = np.full((len(states), len(names)), np.nan, dtype=float)
    for i, d in enumerate(flattened):
        for j, name in enumerate(names):
            if name in d:
                raw[i, j] = d[name]

    keep_names: list[str] = []
    excluded: list[str] = []
    columns: list[np.ndarray] = []

    for j, name in enumerate(names):
        col = raw[:, j]
        valid = np.isfinite(col)
        coverage = float(np.mean(valid)) if len(valid) else 0.0

        if coverage < 0.60:
            excluded.append(f"{name}:LOW_COVERAGE")
            continue

        fill = float(np.nanmean(col))
        col = np.where(valid, col, fill)

        # Constants/shared defaults do not count as independent invariant evidence.
        if float(np.std(col)) < 1e-8:
            excluded.append(f"{name}:CONSTANT_OR_SHARED_DEFAULT")
            continue

        keep_names.append(name)
        columns.append(col)

    if not columns:
        return np.empty((len(states), 0)), [], excluded

    X = np.column_stack(columns)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    X = (X - mu) / np.maximum(sd, 1e-8)
    return X, keep_names, excluded

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)

def slug_axis_part(label: str, max_len: int = 18) -> str:
    s = re.sub(r"[^\w]+", "_", str(label), flags=re.UNICODE).strip("_")
    return (s or "member")[:max_len]

def discover_coordinate_candidates(
    states: list[UtteranceState],
    embeddings: np.ndarray,
    labels: list[str],
    min_members: int = 4,
    min_evr: float = 0.08,
    max_known_corr: float = 0.70,
) -> tuple[list[CoordinateCandidate], list[GapCandidate], list[str], dict[str, Any]]:
    """
    Preserve embedding-supported structure as a candidate missing coordinate
    only when existing observed ATLAS coordinates fail to explain it.
    """
    n = len(states)
    if n < min_members:
        return [], [], [], {
            "status": "INSUFFICIENT_MEMBERS",
            "required_members": min_members,
            "observed_members": n,
        }

    E = normalize_rows(np.asarray(embeddings, dtype=float))
    X, feature_names, excluded = build_observed_feature_matrix(states)

    if E.ndim != 2 or E.shape[0] != n:
        return [], [], excluded, {
            "status": "INVALID_EMBEDDING_MATRIX",
            "states": n,
            "embedding_shape": list(E.shape),
        }

    n_components = min(n - 1, E.shape[1], 8)
    if n_components < 1:
        return [], [], excluded, {"status": "NO_EMBEDDING_COMPONENTS"}

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(E)
    evr = pca.explained_variance_ratio_

    candidates: list[CoordinateCandidate] = []
    gaps: list[GapCandidate] = []

    for k in range(n_components):
        axis = scores[:, k]
        explained_variance = float(evr[k])

        if explained_variance < min_evr:
            continue

        max_corr = 0.0
        best_feature = None

        if X.shape[1]:
            for j, name in enumerate(feature_names):
                corr = abs(safe_corr(axis, X[:, j]))
                if corr > max_corr:
                    max_corr = corr
                    best_feature = name

        novelty = clamp(
            explained_variance
            * (1.0 - max_corr)
            * min(1.0, n / 6.0)
            * 2.5
        )

        if max_corr > max_known_corr:
            continue

        order = np.argsort(axis)
        low_idx = order[: min(2, n)]
        high_idx = order[-min(2, n):][::-1]

        negative_members = [labels[i] for i in low_idx]
        positive_members = [labels[i] for i in high_idx]

        proposed_name = (
            f"AXIS_{slug_axis_part(negative_members[0])}"
            f"__TO__{slug_axis_part(positive_members[0])}"
        )

        language_count = len({s.language for s in states})
        evidence_independence = clamp(
            0.25
            + 0.10 * min(language_count, 5)
            + 0.05 * min(n, 6)
        )

        candidate = CoordinateCandidate(
            id=uid("coord"),
            proposed_name=proposed_name,
            source_axis=f"embedding_PC{k+1}",
            explained_variance=explained_variance,
            max_existing_coordinate_correlation=max_corr,
            novelty_score=novelty,
            positive_members=positive_members,
            negative_members=negative_members,
            supporting_members=n,
            evidence_independence=evidence_independence,
        )
        candidates.append(candidate)

        separation = float(np.max(axis) - np.min(axis))
        persistence = clamp(
            novelty * 0.60
            + explained_variance * 0.25
            + evidence_independence * 0.15
        )

        if novelty >= 0.08 and separation > 1e-6:
            competing = [
                "translation mismatch",
                "embedding-model artifact",
                "shared heuristic encoder bias",
                "context/register mismatch",
                "true missing coordinate",
            ]
            if best_feature:
                competing.append(
                    f"weakly related existing coordinate: {best_feature}"
                )

            gaps.append(
                GapCandidate(
                    id=uid("gap"),
                    gap_type="UNEXPLAINED_CROSS_LANGUAGE_STRUCTURE",
                    location=candidate.proposed_name,
                    magnitude=novelty,
                    persistence=persistence,
                    supporting_members=positive_members + negative_members,
                    competing_explanations=competing,
                    recommended_test=(
                        "Add independent language realizations/contexts; rerun with "
                        "another embedding model; test persistence after replacing "
                        "heuristic coordinates with language-specific observed coordinates."
                    ),
                )
            )

    candidates.sort(key=lambda c: c.novelty_score, reverse=True)
    gaps.sort(key=lambda g: g.persistence, reverse=True)

    diagnostics = {
        "status": "OK",
        "members": n,
        "languages": len({s.language for s in states}),
        "embedding_components_tested": int(n_components),
        "observed_coordinates_used": len(feature_names),
        "coordinates_excluded": len(excluded),
        "discovery_rule": (
            "significant embedding variance + weak correlation with every "
            "registered observed coordinate => candidate missing coordinate"
        ),
    }

    return candidates, gaps, excluded, diagnostics

def detect_term_proposals(
    concept_label: str,
    declared_entries: list[tuple[str, str, str | None]],
    invariant: InvariantCandidate | None,
    residual_df: pd.DataFrame | None = None,
) -> list[TermProposal]:
    proposals: list[TermProposal] = []

    if invariant is None:
        invariant_spaces = []
        invariant_strength = 0.0
    else:
        invariant_spaces = list(invariant.shared_coordinates.keys())
        invariant_strength = invariant.confidence

    observed_lengths = []
    for _, _, txt in declared_entries:
        if txt and txt.strip().upper() not in {"?", "NONE", "GAP", "—", "-"}:
            observed_lengths.append(len(TOKEN_RE.findall(txt)))

    median_len = float(np.median(observed_lengths)) if observed_lengths else 1.0

    for language_name, lang, txt in declared_entries:
        missing = (
            txt is None
            or not txt.strip()
            or txt.strip().upper() in {"?", "NONE", "GAP", "—", "-"}
        )
        token_len = 0 if missing else len(TOKEN_RE.findall(txt))

        explicit_gap = missing
        periphrastic_gap = (
            not missing
            and len(observed_lengths) >= 3
            and median_len <= 1.5
            and token_len >= 3
        )

        if not explicit_gap and not periphrastic_gap:
            continue

        pressure = 0.75 if explicit_gap else 0.48
        pressure = clamp(
            pressure * (0.60 + 0.40 * invariant_strength)
        )

        reason = (
            "Explicitly declared lexical gap for a cross-language invariant candidate."
            if explicit_gap
            else
            "Possible lexicalization gap: this language uses a substantially more "
            "periphrastic realization than peer languages."
        )

        proposals.append(
            TermProposal(
                id=uid("term"),
                language=language_name,
                concept_label=concept_label,
                reason=reason,
                supporting_invariant_spaces=invariant_spaces[:12],
                residual_pressure=pressure,
                proposed_placeholder=f"TERM_GAP_{lang}_{slug_axis_part(concept_label,24)}",
            )
        )

    return proposals

def discover_gaps_and_coordinates(
    states: list[UtteranceState],
    embeddings: np.ndarray,
    labels: list[str],
    concept_label: str = "",
    declared_entries: list[tuple[str, str, str | None]] | None = None,
    invariant: InvariantCandidate | None = None,
    residual_df: pd.DataFrame | None = None,
) -> DiscoveryReport:
    coords, gaps, excluded, diagnostics = discover_coordinate_candidates(
        states=states,
        embeddings=embeddings,
        labels=labels,
    )

    terms = detect_term_proposals(
        concept_label=concept_label or "unnamed_concept",
        declared_entries=declared_entries or [],
        invariant=invariant,
        residual_df=residual_df,
    )

    diagnostics["coordinate_candidates"] = len(coords)
    diagnostics["gap_candidates"] = len(gaps)
    diagnostics["term_proposals"] = len(terms)

    return DiscoveryReport(
        coordinate_candidates=coords,
        gap_candidates=gaps,
        term_proposals=terms,
        excluded_coordinates=excluded,
        diagnostics=diagnostics,
    )

def discovery_coordinate_df(report: DiscoveryReport) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "proposed_name": c.proposed_name,
            "source_axis": c.source_axis,
            "explained_variance": c.explained_variance,
            "max_known_corr": c.max_existing_coordinate_correlation,
            "novelty_score": c.novelty_score,
            "evidence_independence": c.evidence_independence,
            "positive_members": " | ".join(c.positive_members),
            "negative_members": " | ".join(c.negative_members),
            "status": c.status,
        }
        for c in report.coordinate_candidates
    ])

def discovery_gap_df(report: DiscoveryReport) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "gap_type": g.gap_type,
            "location": g.location,
            "magnitude": g.magnitude,
            "persistence": g.persistence,
            "supporting_members": " | ".join(g.supporting_members),
            "competing_explanations": " | ".join(g.competing_explanations),
            "recommended_test": g.recommended_test,
            "status": g.status,
        }
        for g in report.gap_candidates
    ])

def term_proposal_df(report: DiscoveryReport) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "language": t.language,
            "concept": t.concept_label,
            "reason": t.reason,
            "supporting_invariant_spaces": ", ".join(t.supporting_invariant_spaces),
            "residual_pressure": t.residual_pressure,
            "placeholder": t.proposed_placeholder,
            "status": t.status,
        }
        for t in report.term_proposals
    ])


# =============================================================================
# Comparison engine
# =============================================================================

def space_summary(typed_spaces: dict[str, dict[str, float]]) -> dict[str, float]:
    return {s: mean(coords.values()) for s, coords in typed_spaces.items()}

def compare_meaning_frames(a: UtteranceState, b: UtteranceState) -> dict[str, Any]:
    fa = a.meaningFrames[0] if a.meaningFrames else None
    fb = b.meaningFrames[0] if b.meaningFrames else None
    if not fa or not fb:
        return {
            "frame_comparable": False,
            "shared_skeleton": [],
            "structural_differences": [],
        }

    shared = []
    diffs = []

    if fa.frame_type == fb.frame_type:
        shared.append(f"frame_type={fa.frame_type}")
    else:
        diffs.append({"dimension":"frame_type","A":fa.frame_type,"B":fb.frame_type})

    # Experiencer/target labels from canonical form are preserved as report-level signals.
    if fa.predicate == fb.predicate:
        shared.append(f"predicate={fa.predicate}")
    else:
        diffs.append({"dimension":"predicate","A":fa.predicate,"B":fb.predicate})

    if fa.direction == fb.direction:
        shared.append(f"direction={fa.direction}")
    else:
        diffs.append({"dimension":"direction","A":fa.direction,"B":fb.direction})

    if fa.affect_valence is not None and fb.affect_valence is not None:
        if abs(fa.affect_valence-fb.affect_valence) < .15:
            shared.append("affect_valence≈shared")
        else:
            diffs.append({
                "dimension":"affect_valence",
                "A":fa.affect_valence,
                "B":fb.affect_valence,
                "delta":fb.affect_valence-fa.affect_valence
            })

    # Semantic role skeleton
    ra = {r.role for r in fa.roles}
    rb = {r.role for r in fb.roles}
    if ra == rb:
        shared.append("role_skeleton=" + ",".join(sorted(ra)))
    else:
        diffs.append({"dimension":"role_skeleton","A":sorted(ra),"B":sorted(rb)})

    return {
        "frame_comparable": True,
        "shared_skeleton": shared,
        "structural_differences": diffs,
        "canonical_A": fa.canonical_form,
        "canonical_B": fb.canonical_form,
    }

def compare_two_states(a: UtteranceState, b: UtteranceState, embeddings: np.ndarray) -> dict[str, Any]:
    a_sum = space_summary(a.typed_spaces)
    b_sum = space_summary(b.typed_spaces)
    spaces = [s for s in SPACE_ORDER if s in a_sum and s in b_sum]

    delta = {s: b_sum[s] - a_sum[s] for s in spaces}
    abs_delta = {s: abs(delta[s]) for s in spaces}
    similarity = float(cosine_similarity(embeddings[:1], embeddings[1:2])[0,0])

    invariant_spaces = [s for s in spaces if abs_delta[s] <= 0.08]
    divergent_spaces = sorted(spaces, key=lambda s: abs_delta[s], reverse=True)[:8]

    return {
        "embedding_similarity": similarity,
        "space_delta": delta,
        "space_abs_delta": abs_delta,
        "candidate_invariant_spaces": invariant_spaces,
        "largest_divergences": divergent_spaces,
        "meaning_frame_comparison": compare_meaning_frames(a,b),
    }

def align_many(states: list[UtteranceState], embeddings: np.ndarray, labels: list[str]) -> tuple[Alignment, InvariantCandidate, list[Residual], pd.DataFrame]:
    """
    Evidence-aware multilingual alignment.

    Important repair from v4:
      * unknown placeholders do not count as measurements
      * coordinates that are identical because of a shared default are not
        promoted to invariants
      * residuals are calculated coordinate-by-coordinate where possible
    """
    centroid = normalize_rows(np.mean(embeddings, axis=0, keepdims=True))[0]
    emb_coherence = float(np.mean(embeddings @ centroid))

    flattened = [flatten_observed_coordinates(s) for s in states]
    coordinate_names = sorted({k for d in flattened for k in d})

    per_space_values: dict[str, list[float]] = {s: [] for s in SPACE_ORDER}
    shared: dict[str, dict[str, float]] = {}
    residuals: list[Residual] = []
    rows = []

    # Track spaces supported by at least one non-constant observed coordinate.
    space_coherences: dict[str, list[float]] = {s: [] for s in SPACE_ORDER}

    for name in coordinate_names:
        space, coordinate = name.split(".", 1)
        vals = np.array([d.get(name, np.nan) for d in flattened], dtype=float)
        valid = np.isfinite(vals)

        if np.mean(valid) < 0.60:
            continue

        ref = float(np.nanmean(vals))
        filled = np.where(valid, vals, ref)
        dispersion = float(np.std(filled))

        # If all values are identical, we do NOT know whether this is a true
        # invariant or a shared encoder default. Preserve it as non-evidence.
        if dispersion < 1e-8:
            for label, val in zip(labels, vals):
                if np.isfinite(val):
                    rows.append({
                        "member": label,
                        "space": space,
                        "coordinate": coordinate,
                        "value": float(val),
                        "group_mean": ref,
                        "residual": 0.0,
                        "coherence": np.nan,
                        "evidence_class": "CONSTANT_OR_SHARED_DEFAULT",
                    })
            continue

        coherence = clamp(1 - dispersion)
        space_coherences.setdefault(space, []).append(coherence)

        if coherence >= 0.92:
            shared.setdefault(space, {})[coordinate] = ref

        for label, val in zip(labels, vals):
            if np.isfinite(val):
                residual = float(val - ref)
                residuals.append(
                    Residual(
                        member=label,
                        space=space,
                        coordinate=coordinate,
                        value=float(val),
                        reference=ref,
                        residual=residual,
                        status=(
                            "INSIGNIFICANT_RESIDUAL"
                            if abs(residual) < 0.03
                            else "SIGNIFICANT_RESIDUAL"
                        ),
                    )
                )
                rows.append({
                    "member": label,
                    "space": space,
                    "coordinate": coordinate,
                    "value": float(val),
                    "group_mean": ref,
                    "residual": residual,
                    "coherence": coherence,
                    "evidence_class": "OBSERVED_OR_DERIVED",
                })

    per_space_coherence = {}
    for space in SPACE_ORDER:
        vals = space_coherences.get(space, [])
        if vals:
            per_space_coherence[space] = mean(vals)

    alignment = Alignment(
        id=uid("align"),
        members=labels,
        languages=[s.language for s in states],
        embedding_coherence=emb_coherence,
        per_space_coherence=per_space_coherence,
    )

    # Confidence is based only on spaces that had non-constant, observed
    # coordinates. Embedding coherence is capped so it cannot dominate.
    typed_conf = mean(per_space_coherence.values(), 0.0)
    evidence_coverage = clamp(len(per_space_coherence) / max(1, len(SPACE_ORDER)))
    invariant_conf = clamp(
        typed_conf * 0.55
        + min(emb_coherence, 0.95) * 0.20
        + evidence_coverage * 0.25
    )

    invariant = InvariantCandidate(
        id=uid("inv"),
        label="cross-language shared structure",
        support=labels,
        shared_coordinates=shared,
        confidence=invariant_conf,
        status=(
            "CANDIDATE_INVARIANT"
            if shared and evidence_coverage >= 0.15
            else "INSUFFICIENT_INDEPENDENT_EVIDENCE"
        ),
    )

    return alignment, invariant, residuals, pd.DataFrame(rows)


def top_layer_for(states: list[UtteranceState], language_maps: list[LanguageMapState],
                  alignments=None, invariants=None, residuals=None,
                  coordinate_candidates=None, gap_candidates=None, term_proposals=None) -> TopLayerState:
    all_rel = [r for s in states for r in s.relations]
    all_h = [h for s in states for h in s.hyperrelations]
    all_t = [t for s in states for t in s.transformations]
    fibers = [
        {
            "utterance": s.id,
            "token": ts.token.id,
            "surface": ts.token.surface,
            "spaces": list(ts.typed_spaces.keys()),
        }
        for s in states for ts in s.tokens
    ]
    avg_unc = mean(
        u for s in states for u in s.uncertainty.vector.values()
    )
    return TopLayerState(
        id=uid("A"),
        languageMaps=language_maps,
        typedSpaces=SPACE_ORDER,
        fibers=fibers,
        metrics={
            "typed_space_count": len(SPACE_ORDER),
            "utterance_count": len(states),
            "token_count": sum(len(s.tokens) for s in states),
        },
        relations=all_rel,
        hyperrelations=all_h,
        transformations=all_t,
        alignments=alignments or [],
        invariants=invariants or [],
        residuals=residuals or [],
        uncertainty={"aggregate": avg_unc, "ambiguity_preserved": True},
        attention={
            "states": [s.attention for s in states],
            "status": "PROVISIONAL",
        },
        comprehension={
            "states": [s.comprehension for s in states],
            "status": "PROVISIONAL",
        },
        knowledgeDelta={
            "states": [s.knowledgeDelta for s in states],
            "status": "PROVISIONAL",
        },
        provenance=Provenance(source="ATLAS_TOP_LAYER_V10"),
        validationStatus=ValidationState(),
        coordinateCandidates=coordinate_candidates or [],
        gapCandidates=gap_candidates or [],
        termProposals=term_proposals or [],
    )


# =============================================================================
# DataFrame helpers
# =============================================================================

def token_state_df(state: UtteranceState) -> pd.DataFrame:
    rows = []
    for ts in state.tokens:
        for space, coords in ts.typed_spaces.items():
            for coordinate, value in coords.items():
                rows.append({
                    "token_index": ts.token.index_u,
                    "token": ts.token.surface,
                    "lemma": ts.token.lemma,
                    "pos": ts.token.pos,
                    "morph": ts.token.morph,
                    "space": space,
                    "coordinate": coordinate,
                    "value": float(value),
                    "status": ts.validation.validationStatus,
                    "confidence": ts.validation.confidence,
                })
    return pd.DataFrame(rows)

def utterance_space_df(state: UtteranceState) -> pd.DataFrame:
    rows = []
    for space in SPACE_ORDER:
        coords = state.typed_spaces.get(space, {})
        for k,v in coords.items():
            rows.append({
                "space": space,
                "space_name": SPACE_DESCRIPTIONS[space],
                "coordinate": k,
                "value": float(v),
            })
    return pd.DataFrame(rows)

def relation_df(state: UtteranceState) -> pd.DataFrame:
    token_lookup = {ts.token.id: ts.token.surface for ts in state.tokens}
    return pd.DataFrame([
        {
            "source": token_lookup.get(r.source, r.source),
            "target": token_lookup.get(r.target, r.target),
            "relation": r.relation,
            "space": r.space,
            "weight": r.weight,
            "status": r.status,
            "evidence": r.evidence,
        }
        for r in state.relations
    ])

def hyperrelation_df(state: UtteranceState) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": h.id,
            "relation": h.relation,
            "nodes": ", ".join(h.nodes),
            "roles": json.dumps(h.roles, ensure_ascii=False),
            "confidence": h.confidence,
            "status": h.status,
        }
        for h in state.hyperrelations
    ])


# =============================================================================
# Visuals
# =============================================================================

def make_space_bar(state: UtteranceState, title: str) -> go.Figure:
    s = space_summary(state.typed_spaces)
    fig = go.Figure(go.Bar(x=list(s.keys()), y=list(s.values())))
    fig.update_layout(
        template="plotly_dark", height=430, title=title,
        yaxis=dict(range=[0,1]), margin=dict(l=20,r=20,t=50,b=80)
    )
    return fig

def make_comparison_delta(comp: dict[str, Any]) -> go.Figure:
    d = comp["space_delta"]
    fig = go.Figure(go.Bar(x=list(d.keys()), y=list(d.values())))
    fig.update_layout(
        template="plotly_dark",
        height=460,
        title="Typed-space delta: B − A",
        yaxis_title="Δ mean activation",
        margin=dict(l=20,r=20,t=50,b=80),
    )
    return fig

def make_many_heatmap(states: list[UtteranceState], labels: list[str]) -> go.Figure:
    z = np.array([[space_summary(s.typed_spaces)[sp] for sp in SPACE_ORDER] for s in states])
    fig = go.Figure(go.Heatmap(
        z=z, x=SPACE_ORDER, y=labels,
        zmin=0, zmax=1, colorscale="Viridis",
        colorbar=dict(title="mean state")
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(450, 55*len(states)+200),
        title="Members × ATLAS typed spaces"
    )
    return fig

def project_embeddings(embeddings: np.ndarray, dimensions=3) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros((1, dimensions))
    n = min(dimensions, len(embeddings), embeddings.shape[1])
    coords = PCA(n_components=n, random_state=42).fit_transform(embeddings)
    if n < dimensions:
        coords = np.pad(coords, ((0,0),(0,dimensions-n)))
    return coords

def embedding_plot(embeddings: np.ndarray, labels: list[str]) -> go.Figure:
    coords = project_embeddings(embeddings, 3)
    sims = cosine_similarity(embeddings)
    fig = go.Figure()
    ex,ey,ez = [],[],[]
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            if sims[i,j] >= 0.40:
                ex += [coords[i,0],coords[j,0],None]
                ey += [coords[i,1],coords[j,1],None]
                ez += [coords[i,2],coords[j,2],None]
    fig.add_trace(go.Scatter3d(
        x=ex,y=ey,z=ez,mode="lines",
        line=dict(width=2,color="rgba(150,170,190,.28)"),
        hoverinfo="skip",name="embedding similarity"
    ))
    fig.add_trace(go.Scatter3d(
        x=coords[:,0],y=coords[:,1],z=coords[:,2],
        mode="markers+text",text=labels,textposition="top center",
        marker=dict(size=8),name="members"
    ))
    fig.update_layout(
        template="plotly_dark", height=650,
        title="Embedding observation layer",
        scene=dict(xaxis_title="Display 1",yaxis_title="Display 2",zaxis_title="Display 3")
    )
    return fig

def fiber_plot(state: UtteranceState) -> go.Figure:
    fig = go.Figure()
    for ts in state.tokens:
        xs,ys,zs,hover = [],[],[],[]
        for si,sp in enumerate(SPACE_ORDER):
            coords = ts.typed_spaces.get(sp,{})
            if not coords:
                continue
            xs.append(ts.token.index_u)
            ys.append(si)
            zs.append(mean(coords.values()))
            hover.append(
                f"<b>{ts.token.surface}</b><br>{sp}<br>"
                f"mean={mean(coords.values()):.3f}<br>"
                f"status={ts.validation.validationStatus}"
            )
        fig.add_trace(go.Scatter3d(
            x=xs,y=ys,z=zs,mode="lines+markers",
            line=dict(width=3,color="rgba(190,205,220,.25)"),
            marker=dict(size=4,color=[SPACE_COLORS[SPACE_ORDER[int(y)]] for y in ys]),
            customdata=hover,hovertemplate="%{customdata}<extra></extra>",
            name=ts.token.surface
        ))
    fig.update_layout(
        template="plotly_dark", height=780,
        title="Same-token fibers across ATLAS typed spaces",
        scene=dict(
            xaxis=dict(
                title="token",
                tickmode="array",
                tickvals=[ts.token.index_u for ts in state.tokens],
                ticktext=[ts.token.surface for ts in state.tokens]
            ),
            yaxis=dict(
                title="space",
                tickmode="array",
                tickvals=list(range(len(SPACE_ORDER))),
                ticktext=SPACE_ORDER,
            ),
            zaxis=dict(title="mean coordinate activation",range=[0,1])
        )
    )
    return fig



# =============================================================================
# Clean report engine
# =============================================================================

def atlas_report_markdown(
    states: list[UtteranceState],
    language_maps: list[LanguageMapState],
    title: str = "ATLAS Report",
    alignment: Alignment | None = None,
    invariant: InvariantCandidate | None = None,
    residual_df: pd.DataFrame | None = None,
    discovery: DiscoveryReport | None = None,
    embedding_model: str = "",
) -> str:
    """One readable report containing the human-facing ATLAS result."""
    lines = [f"# {title}", ""]

    lines += ["## Executive Summary", ""]
    lines.append(f"- Members: **{len(states)}**")
    lines.append(f"- Languages: **{', '.join(sorted({s.language for s in states}))}**")
    lines.append(f"- Typed spaces: **{len(SPACE_ORDER)}**")
    lines.append(f"- Lexical relation registry: **{len(LEXICAL_RELATION_REGISTRY)} relation types**")
    if alignment:
        lines.append(f"- Embedding coherence: **{alignment.embedding_coherence:.3f}** *(observation layer)*")
    if invariant:
        lines.append(f"- Invariant: **{invariant.status}**, confidence **{invariant.confidence:.3f}**")
    if discovery:
        lines.append(f"- Candidate coordinates: **{len(discovery.coordinate_candidates)}**")
        lines.append(f"- Persistent gaps: **{len(discovery.gap_candidates)}**")
        lines.append(f"- Term proposals: **{len(discovery.term_proposals)}**")
    lines += ["", "> Shared default ≠ invariant. Unknown ≠ agreement. Semantic neighbor ≠ synonym.", ""]

    lines += ["## Inputs", ""]
    for i,s in enumerate(states,1):
        lines += [
            f"### Member {i}",
            f"- Language: `{s.language}`",
            f"- Text: **{s.form}**",
            f"- Tokens: {len(s.tokens)}",
            f"- Validation: `{s.validation.validationStatus}`",
            f"- Confidence: {s.validation.confidence:.3f}",
            ""
        ]

    lines += ["## Active Native Language Analysis", ""]
    for state in states:
        lines.append(f"### {state.form}")
        lines.append(f"- Analyzer: `{state.languageMapAnalyzer or 'unknown'}`")
        if state.nativeMechanicsObserved:
            lines.append(f"- Native mechanics observed: `{json.dumps(state.nativeMechanicsObserved, ensure_ascii=False)}`")
        native_spaces = {}
        for sp in SPACE_ORDER:
            coords = state.typed_spaces.get(sp,{})
            natives = {k:v for k,v in coords.items() if k.startswith("native::")}
            if natives:
                native_spaces[sp] = natives
        if native_spaces:
            lines.append("- Native typed observations:")
            for sp,coords in native_spaces.items():
                lines.append(f"  - `{sp}`: `{json.dumps(coords, ensure_ascii=False)}`")
        else:
            lines.append("- Native typed observations: none.")
        lines.append("")

    lines += ["## Meaning Frames", ""]
    for state in states:
        lines.append(f"### {state.form}")
        if not state.meaningFrames:
            lines.append("- No meaning frame resolved.")
        else:
            lookup = {ts.token.id:ts.token.surface for ts in state.tokens}
            for f in state.meaningFrames:
                lines.append(f"- Frame: **{f.frame_type}**")
                lines.append(f"- Predicate: **{f.predicate}**")
                lines.append(f"- Canonical form: `{f.canonical_form}`")
                lines.append(f"- Polarity: `{f.polarity}`")
                lines.append(f"- Modality: `{f.modality}`")
                if f.affect_type:
                    lines.append(f"- Affect type: `{f.affect_type}`")
                    lines.append(f"- Affect valence: `{f.affect_valence}`")
                    lines.append(f"- Direction: `{f.direction}`")
                for rb in f.roles:
                    lines.append(
                        f"  - {rb.role}: `{lookup.get(rb.filler,rb.filler)}` "
                        f"(confidence={rb.confidence:.2f})"
                    )
        lines.append("")

    lines += ["## Reference / Coreference Graph", ""]
    for state in states:
        lookup = {ts.token.id:ts.token.surface for ts in state.tokens}
        lines.append(f"### {state.form}")
        if state.references:
            for ref in state.references:
                token_names = [lookup.get(t,t) for t in ref.token_ids]
                lines.append(
                    f"- `{ref.label}` person={ref.person}, number={ref.number}, "
                    f"role={ref.discourse_role}, tokens={token_names}"
                )
        else:
            lines.append("- No reference entities resolved.")
        if state.coreference:
            lines.append("- Coreference:")
            for c in state.coreference:
                lines.append(
                    f"  - `{lookup.get(c.source,c.source)}` —**{c.relation}**→ "
                    f"`{lookup.get(c.target,c.target)}` ({c.confidence:.2f})"
                )
        lines.append("")

    lines += ["## Semantic Roles", ""]
    for state in states:
        lookup = {ts.token.id:ts.token.surface for ts in state.tokens}
        lines.append(f"### {state.form}")
        if not state.semanticRoles:
            lines.append("- No semantic-role bindings resolved.")
        else:
            for r in state.semanticRoles:
                lines.append(
                    f"- `{lookup.get(r.predicate,r.predicate)}` / **{r.role}** → "
                    f"`{lookup.get(r.filler,r.filler)}` ({r.confidence:.2f})"
                )
        lines.append("")

    lines += ["## Comprehension Certificate", ""]
    for state in states:
        cert = state.comprehensionCertificate
        lines.append(f"### {state.form}")
        if cert is None:
            lines.append("- No certificate.")
        else:
            lines.append(f"- Predicate resolved: **{cert.predicate_resolved}**")
            lines.append(f"- Subject/experiencer resolved: **{cert.subject_resolved}**")
            lines.append(f"- Object/target resolved: **{cert.object_resolved}**")
            lines.append(f"- Coreference resolved: **{cert.coreference_resolved}**")
            lines.append(f"- Affect target resolved: **{cert.affect_target_resolved}**")
            lines.append(f"- Speaker resolved: **{cert.speaker_resolved}**")
            lines.append(f"- Addressee resolved: **{cert.addressee_resolved}**")
            lines.append(f"- Scope resolved: **{cert.scope_resolved}**")
            lines.append(f"- Certificate confidence: **{cert.confidence:.3f}**")
            if cert.ambiguities:
                lines.append("- Ambiguities: " + "; ".join(cert.ambiguities))
            if cert.unresolved_items:
                lines.append("- Unresolved: " + "; ".join(cert.unresolved_items))
        lines.append("")

    lines += ["## State Δ by Domain", ""]
    for state in states:
        d = state.stateDelta
        lines.append(f"### {state.form}")
        if d:
            lines.append(f"- Knowledge Δ: {d.knowledge:.3f}")
            lines.append(f"- Belief Δ: {d.belief:.3f}")
            lines.append(f"- Affect Δ: {d.affect:.3f}")
            lines.append(f"- Self-state Δ: {d.self_state:.3f}")
            lines.append(f"- Social-relation Δ: {d.social_relation:.3f}")
            lines.append(f"- Pragmatic Δ: {d.pragmatic:.3f}")
            lines.append(f"- Uncertainty: {d.uncertainty:.3f}")
        lines.append("")

    lines += ["## Structural Counterfactual Neighborhood", ""]
    for state in states:
        lines.append(f"### {state.form}")
        if not state.counterfactualNeighbors:
            lines.append("- No structural counterfactuals generated.")
        else:
            for n in state.counterfactualNeighbors:
                lines.append(f"- **{n.operation}** → `{n.text}`")
                lines.append("  - Changed: " + ", ".join(n.changed_dimensions))
                lines.append("  - Expected invariants: " + ", ".join(n.expected_invariants))
                lines.append("  - Expected changes: `" + json.dumps(n.expected_changes, ensure_ascii=False) + "`")
        lines.append("")

    lines += ["## Word-by-Word Lexical Maps", ""]
    for state in states:
        lines += [f"### {state.form}", ""]
        for ts in state.tokens:
            if ts.token.pos == "PUNCT":
                continue
            ln = ts.lexicalNeighborhood
            lines.append(f"#### {ts.token.surface}")
            lines.append(f"- Lemma: `{ts.token.lemma}`")
            lines.append(f"- POS: `{ts.token.pos}`")
            lines.append(f"- Morphology: `{ts.token.morph}`")
            if not ln:
                lines += ["- Lexical neighborhood: unavailable", ""]
                continue
            lines.append(f"- Lexical status: `{ln.status}`")
            lines.append(f"- Senses observed: **{len(ln.senses)}**")
            grouped = {}
            for e in sorted(ln.edges, key=lambda x:x.weight, reverse=True):
                grouped.setdefault(e.relation, []).append(e.target)
            for relation in LEXICAL_RELATION_REGISTRY:
                vals = grouped.get(relation, [])
                if vals:
                    uniq = list(dict.fromkeys(vals))[:12]
                    lines.append(f"- **{relation}**: " + ", ".join(f"`{x}`" for x in uniq))
            missing = [x for x in ln.missing_relation_types if x not in {"HOMOPHONE"}]
            if missing:
                if ts.token.pos in {"PRON","DET","AUX","PART","SCONJ","FUNC"}:
                    lines.append(
                        f"- **Unavailable lexical relation classes**: {len(missing)} "
                        "(suppressed for function/operator word)"
                    )
                else:
                    lines.append(
                        "- **Unobserved / unavailable relation types**: " +
                        ", ".join(missing)
                    )
            if ln.diagnostics:
                reason = ln.diagnostics.get("reason")
                if reason:
                    lines.append(f"- Resource note: {reason}")
            lines.append("")

    lines += ["## Utterance Structural Relations", ""]
    for state in states:
        lookup = {ts.token.id:ts.token.surface for ts in state.tokens}
        lines.append(f"### {state.form}")
        if state.relations:
            for r in state.relations:
                lines.append(
                    f"- `{lookup.get(r.source,r.source)}` —**{r.relation}**→ "
                    f"`{lookup.get(r.target,r.target)}` "
                    f"[{r.space}, w={r.weight:.2f}, {r.status}]"
                )
        else:
            lines.append("- None detected.")
        if state.hyperrelations:
            lines.append("- **Hyperrelations:**")
            for h in state.hyperrelations:
                names = [lookup.get(n,n) for n in h.nodes]
                lines.append(f"  - `{h.relation}`({', '.join(names)}) confidence={h.confidence:.2f}")
        lines.append("")

    lines += ["## Typed-Space State", ""]
    for state in states:
        lines.append(f"### {state.form}")
        sm = space_summary(state.typed_spaces)
        for sp,val in sorted(sm.items(), key=lambda x:x[1], reverse=True):
            lines.append(f"- `{sp}`: {val:.3f}")
        lines.append("")

    lines += ["## Interpretation, Attention, Comprehension, Knowledge Δ", ""]
    for state in states:
        lines.append(f"### {state.form}")
        lines.append(f"- Interpretations: `{json.dumps(state.candidateInterpretations, ensure_ascii=False)}`")
        lines.append(f"- Attention: `{json.dumps(state.attention, ensure_ascii=False)}`")
        lines.append(f"- Comprehension: `{json.dumps(state.comprehension, ensure_ascii=False)}`")
        lines.append(f"- Knowledge Δ: `{json.dumps(state.knowledgeDelta, ensure_ascii=False)}`")
        lines.append("")

    lines += ["## Uncertainty / Validation", ""]
    for state in states:
        lines.append(f"### {state.form}")
        lines.append(f"- Uncertainty vector: `{json.dumps(state.uncertainty.vector, ensure_ascii=False)}`")
        lines.append(f"- Entropy proxy: {state.uncertainty.entropy_proxy:.3f}")
        lines.append(f"- Validation: `{json.dumps(asdict(state.validation), ensure_ascii=False)}`")
        lines.append("")

    if len(states) == 2:
        lines += ["## Pairwise Meaning-Frame Comparison", ""]
        mfcomp = compare_meaning_frames(states[0], states[1])
        lines.append(f"- Comparable: **{mfcomp.get('frame_comparable')}**")
        if mfcomp.get("canonical_A"):
            lines.append(f"- A canonical: `{mfcomp['canonical_A']}`")
            lines.append(f"- B canonical: `{mfcomp['canonical_B']}`")
        if mfcomp.get("shared_skeleton"):
            lines.append("- Shared skeleton:")
            for item in mfcomp["shared_skeleton"]:
                lines.append(f"  - `{item}`")
        if mfcomp.get("structural_differences"):
            lines.append("- Structural differences:")
            for item in mfcomp["structural_differences"]:
                lines.append("  - `" + json.dumps(item, ensure_ascii=False) + "`")
        lines.append("")

    lines += ["## Cross-Member Alignment / Invariants", ""]
    if alignment:
        lines.append(f"- Alignment status: `{alignment.status}`")
        lines.append(f"- Embedding coherence: {alignment.embedding_coherence:.3f}")
        for sp,val in sorted(alignment.per_space_coherence.items(),
                             key=lambda x:x[1], reverse=True):
            lines.append(f"  - `{sp}` coherence: {val:.3f}")
    else:
        lines.append("- No alignment in this run.")
    if invariant:
        lines.append(f"- Invariant status: `{invariant.status}`")
        lines.append(f"- Invariant confidence: {invariant.confidence:.3f}")
        lines.append(f"- Shared spaces: {', '.join(invariant.shared_coordinates) or 'none'}")
    lines.append("")

    lines += ["## Residual Ledger", ""]
    if residual_df is None or residual_df.empty:
        lines.append("- No residual ledger in this run.")
    else:
        rdf = residual_df.copy()
        if "residual" in rdf:
            rdf = rdf.loc[rdf["residual"].abs().sort_values(ascending=False).index]
        for _,r in rdf.head(40).iterrows():
            lines.append(
                f"- `{r.get('member','')}` / `{r.get('space','')}` / "
                f"`{r.get('coordinate','')}` → residual={float(r.get('residual',0)):.3f} "
                f"({r.get('evidence_class','')})"
            )
    lines.append("")

    lines += ["## Structural Discovery", ""]
    if discovery:
        lines.append("### Candidate New Coordinates")
        if discovery.coordinate_candidates:
            for c in discovery.coordinate_candidates:
                lines.append(
                    f"- **{c.proposed_name}**: novelty={c.novelty_score:.3f}, "
                    f"EVR={c.explained_variance:.3f}, known-corr≤{c.max_existing_coordinate_correlation:.3f}"
                )
        else:
            lines.append("- None passed the gate.")
        lines.append("")
        lines.append("### Persistent Gaps")
        if discovery.gap_candidates:
            for g in discovery.gap_candidates:
                lines.append(
                    f"- **{g.location}**: magnitude={g.magnitude:.3f}, "
                    f"persistence={g.persistence:.3f}; test: {g.recommended_test}"
                )
        else:
            lines.append("- None passed the gate.")
        lines.append("")
        lines.append("### New-Term / Lexicalization Proposals")
        if discovery.term_proposals:
            for t in discovery.term_proposals:
                lines.append(
                    f"- **{t.language}** → `{t.proposed_placeholder}`: {t.reason}"
                )
        else:
            lines.append("- None warranted.")
    else:
        lines.append("- Discovery engine was not invoked.")
    lines.append("")

    lines += ["## Language Maps", ""]
    for lm in language_maps:
        lines += [
            f"### {lm.language_name}",
            f"- Family: {lm.family}",
            f"- Analyzer: `{lm.analyzerStatus.get('class','unknown')}`",
            f"- Orthography: `{json.dumps(lm.orthography, ensure_ascii=False)}`",
            f"- Morphology: `{json.dumps(lm.morphology, ensure_ascii=False)}`",
            f"- Syntax: `{json.dumps(lm.syntax, ensure_ascii=False)}`",
            f"- Native mechanics: `{json.dumps(lm.nativeMechanics, ensure_ascii=False)}`",
            f"- Native operators: `{json.dumps(lm.nativeOperators, ensure_ascii=False)}`",
            f"- Native constraints: `{json.dumps(lm.nativeConstraints, ensure_ascii=False)}`",
            f"- Alignment interfaces: `{json.dumps(lm.alignmentInterfaces, ensure_ascii=False)}`",
            f"- Context fibers: {', '.join(lm.contextFibers)}",
            f"- Validation: `{lm.validation.validationStatus}`",
            ""
        ]

    lines += ["## Provenance / Limits", ""]
    lines.append(f"- Embedding model: `{embedding_model or 'not recorded'}`")
    lines.append("- Synonym/antonym/taxonomic/part-whole claims are dictionary-backed only when WordNet/OMW is locally available.")
    lines.append("- Homonymy remains a candidate unless sense separation/etymology supports it.")
    lines.append("- Homophony requires pronunciation evidence and is not inferred from spelling.")
    lines.append("- Cognates, false friends, collocations and idioms require appropriate historical/corpus resources.")
    lines.append("- Candidate coordinates and terms remain hypotheses pending independent validation.")
    lines.append("")
    return "\\n".join(lines)

# =============================================================================
# Streamlit components
# =============================================================================

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top:1rem;padding-bottom:2rem;}
    div[data-testid="stMetric"] {
        background:#0a1018;border:1px solid #263242;border-radius:10px;padding:8px 12px;
    }
    .atlas-law {
        border-left:3px solid #58cbed;padding:9px 12px;background:#070b10;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(
    "Language map → word/token → utterance → comparison/alignment → invariant/residual → ATLAS top layer."
)

with st.sidebar:
    st.header("Experiment")
    mode = st.selectbox("Mode", EXPERIMENT_MODES)

    st.divider()
    st.header("Embedding / NNS")
    offline = st.toggle("Offline hashed embeddings", False)
    model_name = st.text_input("Embedding model", DEFAULT_MODEL)
    hash_dims = st.slider("Offline dimensions",64,1024,384,64)
    use_nns = st.toggle("Train NNS autoencoder", False)
    latent_dim = st.slider("NNS latent dimensions",2,64,16)
    hidden_dim = st.slider("NNS hidden dimensions",16,512,128,16)
    epochs = st.slider("NNS epochs",25,600,150,25)
    lr = st.select_slider(
        "Learning rate",
        options=[1e-4,3e-4,1e-3,3e-3,1e-2],
        value=1e-3,
        format_func=lambda x:f"{x:g}"
    )
    seed = int(st.number_input("Random seed",0,999999,42))

st.markdown(
    """
    <div class="atlas-law">
    Same-object fiber ≠ reasoning path<br>
    Embedding similarity ≠ typed relation<br>
    Cross-language centroid ≠ automatic invariant<br>
    Persistent agreement = invariant candidate; persistent disagreement = residual / candidate axis
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Shared analysis rendering
# =============================================================================

def render_single(state: UtteranceState, lmap: LanguageMapState, embeddings: np.ndarray,
                  embedding_model: str, latent=None, losses=None):
    top = top_layer_for([state],[lmap])

    m = st.columns(8)
    m[0].metric("Language", state.language)
    m[1].metric("Tokens", len(state.tokens))
    m[2].metric("Typed spaces", len(SPACE_ORDER))
    m[3].metric("Meaning frames", len(state.meaningFrames))
    m[4].metric("Relations", len(state.relations))
    m[5].metric("Counterfactuals", len(state.counterfactualNeighbors))
    m[6].metric("Validation", state.validation.validationStatus)
    m[7].metric("Comprehension", f"{state.comprehensionCertificate.confidence:.2f}" if state.comprehensionCertificate else "—")

    tabs = st.tabs([
        "Clean Report","Meaning Frame","Reference + Roles","Counterfactuals",
        "Top Layer","Language Map","Utterance State","Each Word / Token",
        "Lexical Relations","Fibers","Relations + Hyperrelations",
        "Uncertainty / Validation","Embedding / NNS","Export"
    ])

    with tabs[0]:
        clean_report = atlas_report_markdown(
            [state],[lmap],
            title=f"ATLAS Report — {state.form}",
            embedding_model=embedding_model,
        )
        st.markdown(clean_report)
        st.download_button(
            "Download clean report",
            clean_report.encode("utf-8"),
            "atlas_clean_report.md","text/markdown",
            use_container_width=True
        )

    with tabs[1]:
        st.subheader("Meaning Frame")
        if not state.meaningFrames:
            st.info("No meaning frame resolved.")
        else:
            st.dataframe(pd.DataFrame([{
                "frame_type":f.frame_type,
                "predicate":f.predicate,
                "canonical_form":f.canonical_form,
                "polarity":f.polarity,
                "modality":f.modality,
                "affect_type":f.affect_type,
                "affect_valence":f.affect_valence,
                "direction":f.direction,
                "confidence":f.confidence,
                "status":f.status,
            } for f in state.meaningFrames]),hide_index=True,use_container_width=True)
            for f in state.meaningFrames:
                st.code(f.canonical_form)
        st.markdown("#### Comprehension certificate")
        st.json(asdict(state.comprehensionCertificate) if state.comprehensionCertificate else {})

    with tabs[2]:
        lookup = {ts.token.id:ts.token.surface for ts in state.tokens}
        st.markdown("#### Reference entities")
        st.dataframe(pd.DataFrame([asdict(r) for r in state.references]),hide_index=True,use_container_width=True)
        st.markdown("#### Coreference")
        st.dataframe(pd.DataFrame([{
            "source":lookup.get(c.source,c.source),
            "target":lookup.get(c.target,c.target),
            "relation":c.relation,
            "confidence":c.confidence,
            "evidence":c.evidence
        } for c in state.coreference]),hide_index=True,use_container_width=True)
        st.markdown("#### Semantic roles")
        st.dataframe(pd.DataFrame([{
            "predicate":lookup.get(r.predicate,r.predicate),
            "role":r.role,
            "filler":lookup.get(r.filler,r.filler),
            "confidence":r.confidence,
            "evidence":r.evidence,
        } for r in state.semanticRoles]),hide_index=True,use_container_width=True)
        st.markdown("#### State Δ")
        st.json(asdict(state.stateDelta) if state.stateDelta else {})

    with tabs[3]:
        st.subheader("Structural Counterfactual Neighborhood")
        if not state.counterfactualNeighbors:
            st.info("No structural counterfactuals generated.")
        else:
            st.dataframe(pd.DataFrame([{
                "text":n.text,
                "operation":n.operation,
                "changed_dimensions":", ".join(n.changed_dimensions),
                "expected_invariants":", ".join(n.expected_invariants),
                "expected_changes":json.dumps(n.expected_changes,ensure_ascii=False),
            } for n in state.counterfactualNeighbors]),hide_index=True,use_container_width=True,height=460)
            st.info("Lexical antonymy and grammatical negation are deliberately separate transformations.")

    with tabs[4]:
        st.subheader("ATLAS top-layer state")
        st.json(asdict(top), expanded=False)

    with tabs[5]:
        st.subheader(f"Active Native Language Map — {lmap.language_name}")
        st.caption(
            "This map is an active analysis interface. Its coordinates describe native "
            "language mechanics; missing coordinates remain unobserved rather than defaulting to 0.45/0.50."
        )
        st.plotly_chart(native_language_map_graph(lmap),use_container_width=True)
        map_rows=[]
        for sp,vals in lmap.typed_spaces.items():
            for coord,val in vals.items():
                map_rows.append({"space":sp,"native_coordinate":coord,"value":val})
        st.dataframe(pd.DataFrame(map_rows),hide_index=True,use_container_width=True,height=520)
        cma,cmb=st.columns(2)
        with cma:
            st.markdown("#### Native mechanics / operators")
            st.json({
                "mechanics":lmap.nativeMechanics,
                "operators":lmap.nativeOperators,
                "constraints":lmap.nativeConstraints,
            })
        with cmb:
            st.markdown("#### Metrics / alignment interfaces")
            st.json({
                "metrics":lmap.nativeMetrics,
                "alignmentInterfaces":lmap.alignmentInterfaces,
                "analyzerStatus":lmap.analyzerStatus,
                "observationCoverage":lmap.observationCoverage,
            })

    with tabs[6]:
        st.subheader("Full utterance state")
        st.code(state.form)
        st.plotly_chart(make_space_bar(state, "Utterance typed-space state"), use_container_width=True)
        st.dataframe(utterance_space_df(state),hide_index=True,use_container_width=True,height=560)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("#### Candidate interpretations")
            st.json(state.candidateInterpretations)
        with c2:
            st.markdown("#### Structural attention")
            st.json(state.attention)
        with c3:
            st.markdown("#### Comprehension / multidomain Δ")
            st.json({
                "comprehension":state.comprehension,
                "certificate":asdict(state.comprehensionCertificate) if state.comprehensionCertificate else {},
                "stateDelta":asdict(state.stateDelta) if state.stateDelta else {},
            })

    with tabs[7]:
        st.subheader("State of each word / token")
        token_choices = [f"{ts.token.id}: {ts.token.surface}" for ts in state.tokens]
        sel = st.selectbox("Inspect token",token_choices)
        idx = token_choices.index(sel)
        ts = state.tokens[idx]
        a,b = st.columns([1,2])
        with a:
            st.json(asdict(ts.token))
            st.markdown("#### Uncertainty")
            st.json(asdict(ts.uncertainty))
            st.markdown("#### Provenance / validation")
            st.json({"provenance":asdict(ts.provenance),"validation":asdict(ts.validation)})
        with b:
            for sp in SPACE_ORDER:
                with st.expander(f"{sp} — {SPACE_DESCRIPTIONS[sp]}", expanded=sp in {"ROLE","REF","SEM","LOG","EPI","AFFECT","SOC_REL","SELF","COMP"}):
                    df = pd.DataFrame({
                        "coordinate":list(ts.typed_spaces[sp].keys()),
                        "value":list(ts.typed_spaces[sp].values())
                    })
                    st.dataframe(df,hide_index=True,use_container_width=True)
        st.markdown("#### All token-state rows")
        st.dataframe(token_state_df(state),hide_index=True,use_container_width=True,height=560)

    with tabs[8]:
        st.subheader("ATLAS Lexical Relation Map")
        st.plotly_chart(lexical_graph(state),use_container_width=True)
        lcov = lexical_coverage_df(state)
        ledges = lexical_edges_df(state)
        lsenses = lexical_senses_df(state)
        st.markdown("#### Relation coverage by word")
        st.dataframe(lcov,hide_index=True,use_container_width=True,height=420)
        st.markdown("#### Typed lexical edges")
        if ledges.empty:
            st.info("No dictionary-backed lexical relation edges were available.")
        else:
            st.dataframe(ledges,hide_index=True,use_container_width=True,height=560)
        st.markdown("#### Sense inventory")
        if lsenses.empty:
            st.info("No dictionary-backed sense inventory was available.")
        else:
            st.dataframe(lsenses,hide_index=True,use_container_width=True,height=420)

    with tabs[9]:
        st.plotly_chart(fiber_plot(state),use_container_width=True,config={"scrollZoom":True,"displaylogo":False})
        st.info("The vertical line is an identity-preserving fiber, not a reasoning path.")

    with tabs[10]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Relations")
            st.dataframe(relation_df(state),hide_index=True,use_container_width=True)
        with c2:
            st.markdown("#### Hyperrelations")
            st.dataframe(hyperrelation_df(state),hide_index=True,use_container_width=True)
        st.markdown("#### Transformations")
        st.dataframe(pd.DataFrame([asdict(t) for t in state.transformations]),hide_index=True,use_container_width=True)

    with tabs[11]:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Utterance uncertainty")
            st.json(asdict(state.uncertainty))
        with c2:
            st.markdown("#### Validation")
            st.json(asdict(state.validation))

    with tabs[12]:
        labels = [ts.token.surface for ts in state.tokens if ts.token.pos != "PUNCT"]
        plot_emb = embeddings[:len(labels)] if len(embeddings) >= len(labels) else embeddings
        st.plotly_chart(embedding_plot(plot_emb,labels[:len(plot_emb)]),use_container_width=True)
        st.caption(f"Embedding model: {embedding_model}")
        if latent is not None:
            st.metric("Latent dimensions",latent.shape[1])
            if losses:
                fig=go.Figure(go.Scatter(x=np.arange(1,len(losses)+1),y=losses,mode="lines"))
                fig.update_layout(template="plotly_dark",title="NNS reconstruction loss",height=350)
                st.plotly_chart(fig,use_container_width=True)

    with tabs[13]:
        export = {
            "top_layer":asdict(top),
            "language_map":asdict(lmap),
            "utterance_state":asdict(state),
            "meaning_frames":[asdict(x) for x in state.meaningFrames],
            "references":[asdict(x) for x in state.references],
            "coreference":[asdict(x) for x in state.coreference],
            "semantic_roles":[asdict(x) for x in state.semanticRoles],
            "comprehension_certificate":asdict(state.comprehensionCertificate) if state.comprehensionCertificate else None,
            "state_delta":asdict(state.stateDelta) if state.stateDelta else None,
            "counterfactual_neighbors":[asdict(x) for x in state.counterfactualNeighbors],
            "lexical_relations": lexical_edges_df(state).to_dict(orient="records"),
            "lexical_senses": lexical_senses_df(state).to_dict(orient="records"),
        }
        payload=json.dumps(export,ensure_ascii=False,indent=2).encode("utf-8")
        st.download_button("Download ATLAS JSON",payload,"atlas_state.json","application/json",use_container_width=True)
        st.download_button(
            "Download token states CSV",
            token_state_df(state).to_csv(index=False).encode("utf-8"),
            "atlas_token_states.csv","text/csv",use_container_width=True
        )
        st.download_button(
            "Download lexical relations CSV",
            lexical_edges_df(state).to_csv(index=False).encode("utf-8"),
            "atlas_lexical_relations.csv","text/csv",use_container_width=True
        )
        report_payload = atlas_report_markdown(
            [state],[lmap],
            title=f"ATLAS Report — {state.form}",
            embedding_model=embedding_model
        )
        st.download_button(
            "Download complete clean report",
            report_payload.encode("utf-8"),
            "atlas_complete_report.md","text/markdown",use_container_width=True
        )

# =============================================================================
# Mode routing
# =============================================================================

if mode == "Single word / utterance + each token":
    c1,c2 = st.columns([1,1])
    with c1:
        lang_name = st.selectbox("Language", list(LANGUAGES.keys()), key="single_lang")
    with c2:
        context = st.text_input("Optional context", "", key="single_context")
    text = st.text_area(
        "Enter a word, phrase, or full utterance",
        "I believe the model may understand the evidence, but I do not yet know whether its conclusion is true.",
        height=150,
    )
    run = st.button("Construct full ATLAS state",type="primary",use_container_width=True)

    if run:
        lang = LANGUAGES[lang_name]
        state = make_utterance_state(text.strip(),lang)
        lmap = language_map_profile(lang,lang_name)
        token_texts = [f"{ts.token.surface} [language={lang}] context: {context or text}" for ts in state.tokens]
        embeddings, emb_model = embed(token_texts,model_name,offline,hash_dims)

        latent=recon=losses=None
        if use_nns and len(embeddings)>=2:
            latent,recon,losses=train_autoencoder(embeddings,latent_dim,hidden_dim,epochs,lr,seed)

        st.session_state["single_result"]=(state,lmap,embeddings,emb_model,latent,losses)

    if "single_result" in st.session_state:
        render_single(*st.session_state["single_result"])


elif mode in {"Word ↔ Word","Utterance ↔ Utterance"}:
    word_mode = mode.startswith("Word")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("A")
        la_name=st.selectbox("Language A",list(LANGUAGES.keys()),key="pair_la")
        a=st.text_area("Word A" if word_mode else "Utterance A","wisdom" if word_mode else "I love myself.",height=120)
    with c2:
        st.subheader("B")
        lb_name=st.selectbox("Language B",list(LANGUAGES.keys()),index=0,key="pair_lb")
        b=st.text_area("Word B" if word_mode else "Utterance B","حكمة" if word_mode else "I hate you.",height=120)

    if st.button("Compare A ↔ B",type="primary",use_container_width=True):
        la,lb=LANGUAGES[la_name],LANGUAGES[lb_name]
        sa,sb=make_utterance_state(a.strip(),la),make_utterance_state(b.strip(),lb)
        embs,emodel=embed([a,b],model_name,offline,hash_dims)
        comp=compare_two_states(sa,sb,embs)
        lm_a,lm_b=language_map_profile(la,la_name),language_map_profile(lb,lb_name)
        align,inv,resid,resdf=align_many([sa,sb],embs,[f"A:{a}",f"B:{b}"])
        top=top_layer_for([sa,sb],[lm_a,lm_b],[align],[inv],resid)
        st.session_state["pair_result"]=(sa,sb,embs,emodel,comp,align,inv,resdf,top)

    if "pair_result" in st.session_state:
        sa,sb,embs,emodel,comp,align,inv,resdf,top=st.session_state["pair_result"]
        m=st.columns(4)
        m[0].metric("Embedding similarity",f"{comp['embedding_similarity']:.3f}")
        m[1].metric("Invariant spaces",len(comp["candidate_invariant_spaces"]))
        m[2].metric("Largest divergence",comp["largest_divergences"][0] if comp["largest_divergences"] else "—")
        m[3].metric("Alignment confidence",f"{inv.confidence:.3f}")

        tabs=st.tabs([
            "Clean Pair Report","Meaning Frames","Typed-space comparison",
            "A state","B state","Alignment / invariant","Residuals","Top layer"
        ])

        with tabs[0]:
            lm_a = language_map_profile(sa.language, next((n for n,c in LANGUAGES.items() if c==sa.language), sa.language))
            lm_b = language_map_profile(sb.language, next((n for n,c in LANGUAGES.items() if c==sb.language), sb.language))
            pair_report = atlas_report_markdown(
                [sa,sb],[lm_a,lm_b],
                title="ATLAS Pair Comparison Report",
                alignment=align,invariant=inv,residual_df=resdf,
                embedding_model=emodel
            )
            st.markdown(pair_report)
            st.download_button(
                "Download pair report", pair_report.encode("utf-8"),
                "atlas_pair_report.md","text/markdown",use_container_width=True
            )

        with tabs[1]:
            mf = comp["meaning_frame_comparison"]
            st.json(mf)
            ca,cb = st.columns(2)
            with ca:
                st.markdown("#### A")
                for f in sa.meaningFrames:
                    st.code(f.canonical_form)
            with cb:
                st.markdown("#### B")
                for f in sb.meaningFrames:
                    st.code(f.canonical_form)

        with tabs[2]:
            st.plotly_chart(make_comparison_delta(comp),use_container_width=True)
            st.dataframe(pd.DataFrame([
                {"space":s,"A":space_summary(sa.typed_spaces)[s],"B":space_summary(sb.typed_spaces)[s],
                 "delta":comp["space_delta"][s],"abs_delta":comp["space_abs_delta"][s],
                 "candidate_invariant":s in comp["candidate_invariant_spaces"]}
                for s in SPACE_ORDER
            ]),hide_index=True,use_container_width=True)

        with tabs[3]:
            st.plotly_chart(make_space_bar(sa,"A"),use_container_width=True)
            st.json({
                "meaningFrames":[asdict(x) for x in sa.meaningFrames],
                "reference":[asdict(x) for x in sa.references],
                "roles":[asdict(x) for x in sa.semanticRoles],
                "stateDelta":asdict(sa.stateDelta) if sa.stateDelta else None,
            })

        with tabs[4]:
            st.plotly_chart(make_space_bar(sb,"B"),use_container_width=True)
            st.json({
                "meaningFrames":[asdict(x) for x in sb.meaningFrames],
                "reference":[asdict(x) for x in sb.references],
                "roles":[asdict(x) for x in sb.semanticRoles],
                "stateDelta":asdict(sb.stateDelta) if sb.stateDelta else None,
            })

        with tabs[5]:
            st.json({"alignment":asdict(align),"invariant":asdict(inv)})
            st.plotly_chart(embedding_plot(embs,["A","B"]),use_container_width=True)

        with tabs[6]:
            st.dataframe(resdf,hide_index=True,use_container_width=True)

        with tabs[7]:
            st.json(asdict(top),expanded=False)



elif mode in {"Word → Many Languages","Utterance → Many Languages"}:
    word_mode=mode.startswith("Word")
    st.markdown(
        "Enter one language realization per line using `Language Name | text`. "
        "For arbitrary concepts/utterances, ATLAS should compare supplied realizations rather than silently invent translations."
    )
    default = (
        "English | wisdom\nLatin | sapientia\nMandarin Chinese | 智慧\nArabic | حكمة\nFarsi / Persian | خرد"
        if word_mode else
        "English | I believe the evidence may be sufficient.\nFarsi / Persian | من فکر می‌کنم شواهد ممکن است کافی باشد.\nArabic | أعتقد أن الدليل قد يكون كافياً.\nMandarin Chinese | 我认为证据可能足够。"
    )
    entries=st.text_area("Multilingual realizations",default,height=220)
    gloss=st.text_input("Concept / comparison label","wisdom" if word_mode else "epistemic utterance")

    if st.button("Triangulate across languages",type="primary",use_container_width=True):
        declared_entries=[]
        observed=[]
        for line in entries.splitlines():
            if "|" not in line:
                continue
            name,entry_text=[x.strip() for x in line.split("|",1)]
            if name not in LANGUAGES:
                continue
            lang=LANGUAGES[name]
            declared_entries.append((name,lang,entry_text))
            if entry_text and entry_text.strip().upper() not in {"?","NONE","GAP","—","-"}:
                observed.append((name,lang,entry_text))

        if len(observed)<2:
            st.error("Provide at least two observed language realizations. Missing languages may be marked with `?`.")
        else:
            states=[make_utterance_state(entry_text,lang) for name,lang,entry_text in observed]
            labels=[f"{name}: {entry_text}" for name,lang,entry_text in observed]
            texts=[entry_text for name,lang,entry_text in observed]
            embs,emodel=embed(texts,model_name,offline,hash_dims)
            maps=[language_map_profile(lang,name) for name,lang,entry_text in observed]
            align,inv,resid,resdf=align_many(states,embs,labels)

            discovery=discover_gaps_and_coordinates(
                states=states,
                embeddings=embs,
                labels=labels,
                concept_label=gloss,
                declared_entries=declared_entries,
                invariant=inv,
                residual_df=resdf,
            )

            top=top_layer_for(
                states,maps,[align],[inv],resid,
                coordinate_candidates=discovery.coordinate_candidates,
                gap_candidates=discovery.gap_candidates,
                term_proposals=discovery.term_proposals,
            )
            st.session_state["many_result"]=(states,labels,embs,emodel,align,inv,resdf,top,discovery)

    if "many_result" in st.session_state:
        states,labels,embs,emodel,align,inv,resdf,top,discovery=st.session_state["many_result"]
        m=st.columns(4)
        m[0].metric("Languages",len(states))
        m[1].metric("Embedding coherence",f"{align.embedding_coherence:.3f}")
        m[2].metric("Shared typed spaces",len(inv.shared_coordinates))
        m[3].metric("Invariant confidence",f"{inv.confidence:.3f}")

        tabs=st.tabs([
            "Clean Report","Triangulation","Member states","Lexical Relations","Invariant","Residuals",
            "New Coordinates","Persistent Gaps","Term Proposals","Pairwise similarity","Top layer"
        ])
        with tabs[0]:
            report_maps = [
                language_map_profile(
                    s.language,
                    next((name for name,code in LANGUAGES.items() if code == s.language), s.language)
                ) for s in states
            ]
            report_md = atlas_report_markdown(
                states,report_maps,
                title="ATLAS Multilingual Triangulation Report",
                alignment=align,invariant=inv,residual_df=resdf,
                discovery=discovery,embedding_model=emodel
            )
            st.markdown(report_md)
            st.download_button(
                "Download complete multilingual report",
                report_md.encode("utf-8"),
                "atlas_multilingual_report.md","text/markdown",
                use_container_width=True
            )

        with tabs[1]:
            st.plotly_chart(make_many_heatmap(states,labels),use_container_width=True)
            st.plotly_chart(embedding_plot(embs,labels),use_container_width=True)
        with tabs[2]:
            member=st.selectbox("Inspect member",labels)
            idx=labels.index(member)
            st.plotly_chart(make_space_bar(states[idx],member),use_container_width=True)
            st.dataframe(token_state_df(states[idx]),hide_index=True,use_container_width=True)
        with tabs[3]:
            lexical_member = st.selectbox("Inspect lexical member",labels,key="many_lex_member")
            li = labels.index(lexical_member)
            st.plotly_chart(lexical_graph(states[li]),use_container_width=True)
            st.dataframe(lexical_coverage_df(states[li]),hide_index=True,use_container_width=True)
            le = lexical_edges_df(states[li])
            if not le.empty:
                st.dataframe(le,hide_index=True,use_container_width=True,height=520)

        with tabs[4]:
            st.json({"alignment":asdict(align),"invariant":asdict(inv)})
            inv_df=pd.DataFrame([
                {"space":sp,**vals}
                for sp,vals in inv.shared_coordinates.items()
            ])
            st.dataframe(inv_df,hide_index=True,use_container_width=True)
        with tabs[5]:
            st.dataframe(resdf.sort_values("residual",key=lambda s:s.abs(),ascending=False),
                         hide_index=True,use_container_width=True,height=600)
        with tabs[6]:
            st.markdown("### Candidate new coordinates")
            st.caption(
                "A candidate is emitted only when an embedding-supported axis carries "
                "meaningful variance yet correlates weakly with every currently observed "
                "ATLAS coordinate. It is a discovery target, not an accepted semantic axis."
            )
            coord_df=discovery_coordinate_df(discovery)
            if coord_df.empty:
                st.info("No candidate new coordinate passed the current discovery gate.")
            else:
                st.dataframe(coord_df,hide_index=True,use_container_width=True)
            with st.expander("Discovery diagnostics"):
                st.json(discovery.diagnostics)
                st.markdown("#### Excluded coordinates")
                st.code("\n".join(discovery.excluded_coordinates[:250]) or "None")

        with tabs[7]:
            st.markdown("### Persistent gap candidates")
            gap_df=discovery_gap_df(discovery)
            if gap_df.empty:
                st.info("No persistent gap candidate passed the current gate.")
            else:
                st.dataframe(gap_df,hide_index=True,use_container_width=True)

        with tabs[8]:
            st.markdown("### Where a new term may be warranted")
            st.caption(
                "Use `Language Name | ?` to explicitly mark a lexical gap. "
                "ATLAS proposes only a placeholder identifier; it does not invent a "
                "target-language word without morphology/phonology/register validation."
            )
            term_df=term_proposal_df(discovery)
            if term_df.empty:
                st.info("No lexicalization gap currently warrants a term proposal.")
            else:
                st.dataframe(term_df,hide_index=True,use_container_width=True)

        with tabs[9]:
            sim=cosine_similarity(embs)
            st.dataframe(pd.DataFrame(sim,index=labels,columns=labels),use_container_width=True)

        with tabs[10]:
            st.json(asdict(top),expanded=False)



elif mode == "Coordinate / Gap Discovery":
    st.subheader("ATLAS Structural Discovery Engine")
    st.write(
        "Supply several independent realizations or neighboring concepts. "
        "The engine tests whether the currently registered coordinates explain "
        "their observed embedding structure. Unexplained persistent structure is "
        "preserved as a candidate coordinate/gap instead of being forced into an "
        "existing space."
    )

    discovery_label = st.text_input(
        "Concept / experiment label",
        "wisdom",
        key="discovery_label",
    )

    discovery_entries = st.text_area(
        "Language realizations — use `Language Name | text`; mark a lexical gap with `?`",
        "English | wisdom\n"
        "Latin | sapientia\n"
        "Mandarin Chinese | 智慧\n"
        "Arabic | حكمة\n"
        "Farsi / Persian | خرد\n"
        "Turkish | bilgelik\n"
        "Japanese | 知恵\n"
        "Korean | 지혜",
        height=260,
        key="discovery_entries",
    )

    c1,c2,c3 = st.columns(3)
    min_members = c1.slider("Minimum members",3,10,4)
    min_evr = c2.slider("Minimum unexplained axis variance",0.02,0.40,0.08,0.01)
    max_corr = c3.slider("Maximum correlation with known coordinates",0.20,0.95,0.70,0.05)

    if st.button("Run structural discovery",type="primary",use_container_width=True):
        declared=[]
        observed=[]

        for line in discovery_entries.splitlines():
            if "|" not in line:
                continue
            name,entry_text=[x.strip() for x in line.split("|",1)]
            if name not in LANGUAGES:
                continue
            lang=LANGUAGES[name]
            declared.append((name,lang,entry_text))
            if entry_text and entry_text.strip().upper() not in {"?","NONE","GAP","—","-"}:
                observed.append((name,lang,entry_text))

        if len(observed) < min_members:
            st.error(f"Need at least {min_members} observed members for coordinate discovery.")
        else:
            states=[make_utterance_state(entry_text,lang) for name,lang,entry_text in observed]
            labels=[f"{name}: {entry_text}" for name,lang,entry_text in observed]
            embs,emodel=embed([x[2] for x in observed],model_name,offline,hash_dims)
            maps=[language_map_profile(lang,name) for name,lang,entry_text in observed]
            align,inv,resid,resdf=align_many(states,embs,labels)

            coords,gaps,excluded,diagnostics=discover_coordinate_candidates(
                states=states,
                embeddings=embs,
                labels=labels,
                min_members=min_members,
                min_evr=min_evr,
                max_known_corr=max_corr,
            )
            terms=detect_term_proposals(
                concept_label=discovery_label,
                declared_entries=declared,
                invariant=inv,
                residual_df=resdf,
            )
            report=DiscoveryReport(
                coordinate_candidates=coords,
                gap_candidates=gaps,
                term_proposals=terms,
                excluded_coordinates=excluded,
                diagnostics=diagnostics,
            )
            top=top_layer_for(
                states,maps,[align],[inv],resid,
                coordinate_candidates=coords,
                gap_candidates=gaps,
                term_proposals=terms,
            )
            st.session_state["discovery_result"]=(states,labels,embs,emodel,align,inv,resdf,report,top)

    if "discovery_result" in st.session_state:
        states,labels,embs,emodel,align,inv,resdf,report,top=st.session_state["discovery_result"]

        m=st.columns(6)
        m[0].metric("Members",len(states))
        m[1].metric("Languages",len({s.language for s in states}))
        m[2].metric("Embedding coherence",f"{align.embedding_coherence:.3f}")
        m[3].metric("New coordinate candidates",len(report.coordinate_candidates))
        m[4].metric("Persistent gaps",len(report.gap_candidates))
        m[5].metric("Term proposals",len(report.term_proposals))

        tabs=st.tabs([
            "Discovery Overview","Candidate Coordinates","Persistent Gaps",
            "Term Proposals","Observed Structure","Residual Ledger","Top Layer"
        ])

        with tabs[0]:
            st.plotly_chart(make_many_heatmap(states,labels),use_container_width=True)
            st.plotly_chart(embedding_plot(embs,labels),use_container_width=True)
            st.json(report.diagnostics)
            st.markdown(
                """
                **Discovery law**

                `new coordinate candidate = persistent observed structure`
                `− structure explainable by registered coordinates`

                A candidate must then survive independent languages, contexts,
                alternate embedding models, and falsification before registry admission.
                """
            )

        with tabs[1]:
            df=discovery_coordinate_df(report)
            if df.empty:
                st.info("No candidate coordinate passed the discovery gate.")
            else:
                st.dataframe(df,hide_index=True,use_container_width=True,height=520)

        with tabs[2]:
            df=discovery_gap_df(report)
            if df.empty:
                st.info("No persistent gap candidate passed the gate.")
            else:
                st.dataframe(df,hide_index=True,use_container_width=True,height=520)

        with tabs[3]:
            df=term_proposal_df(report)
            if df.empty:
                st.info(
                    "No term proposal warranted. Mark a missing realization with "
                    "`Language Name | ?` to test an explicit lexical gap."
                )
            else:
                st.dataframe(df,hide_index=True,use_container_width=True)
            st.warning(
                "ATLAS proposes a placeholder coordinate/term requirement here, "
                "not a fabricated word. Actual lexical engineering must occur in the "
                "target language map."
            )

        with tabs[4]:
            st.json({"alignment":asdict(align),"invariant":asdict(inv)})
            st.markdown("#### Coordinates excluded from discovery geometry")
            st.code("\\n".join(report.excluded_coordinates[:300]) or "None")

        with tabs[5]:
            if resdf.empty:
                st.info("No residual rows.")
            else:
                st.dataframe(
                    resdf.sort_values(
                        "residual",
                        key=lambda s:s.abs(),
                        ascending=False
                    ),
                    hide_index=True,use_container_width=True,height=620
                )

        with tabs[6]:
            st.json(asdict(top),expanded=False)


elif mode == "Language Intelligence Benchmark":
    st.subheader("ATLAS Language Intelligence Benchmark")
    st.write(
        "Provide a corpus of test utterances grouped by ATLAS language map. "
        "ATLAS will construct every utterance independently, aggregate per-language "
        "intelligence profiles, compare languages, build a failure/falsification ledger, "
        "and optionally run structural coordinate/gap discovery across the benchmark."
    )

    default_benchmark = """[English]
I love myself.
I hate you.
I believe the evidence may support this conclusion.
I do not yet know whether it is true.
The result could change if new information appears.

[Farsi / Persian]
من خودم را دوست دارم.
من از تو متنفرم.
من باور دارم که شواهد ممکن است از این نتیجه پشتیبانی کنند.

[Arabic]
أنا أحب نفسي.
أنا أكرهك.
أعتقد أن الأدلة قد تدعم هذه النتيجة.

[Mandarin Chinese]
我爱我自己。
我恨你。
我认为证据可能支持这个结论。

[Turkish]
Kendimi seviyorum.
Senden nefret ediyorum.
Kanıtların bu sonucu destekleyebileceğine inanıyorum.
"""

    raw_benchmark = st.text_area(
        "Benchmark corpus",
        default_benchmark,
        height=460,
        help=(
            "Use [Language Name] section headers followed by utterances, "
            "or one-line `Language Name | utterance` format."
        )
    )

    c1,c2,c3,c4 = st.columns(4)
    run_discovery = c1.toggle("Run coordinate/gap discovery", True)
    min_discovery_members = c2.slider("Discovery minimum members",3,12,6)
    min_discovery_evr = c3.slider("Discovery min EVR",0.02,0.35,0.07,0.01)
    discovery_max_corr = c4.slider("Discovery max known corr",0.20,0.95,0.70,0.05)

    if st.button("Run ATLAS Intelligence Benchmark",type="primary",use_container_width=True):
        parsed = parse_language_benchmark_input(raw_benchmark)

        if not parsed:
            st.error("No valid benchmark utterances were parsed.")
        else:
            grouped_rows: dict[str,list[tuple[str,str,str]]] = {}
            language_codes = {}
            for lang_name,lang_code,utt in parsed:
                grouped_rows.setdefault(lang_name,[]).append((lang_name,lang_code,utt))
                language_codes[lang_name] = lang_code

            grouped_states: dict[str,list[UtteranceState]] = {}
            all_states = []
            all_labels = []
            all_texts = []

            progress = st.progress(0.0)
            total = len(parsed)
            done = 0

            for lang_name, rows in grouped_rows.items():
                grouped_states[lang_name] = []
                for _,lang_code,utt in rows:
                    state = make_utterance_state(utt,lang_code)
                    grouped_states[lang_name].append(state)
                    all_states.append(state)
                    all_labels.append(f"{lang_name}: {utt}")
                    all_texts.append(utt)
                    done += 1
                    progress.progress(done/max(1,total))

            embeddings, emodel = embed(
                all_texts,model_name,offline,hash_dims
            )

            discovery = None
            if run_discovery and len(all_states) >= min_discovery_members:
                coords,gaps,excluded,diagnostics = discover_coordinate_candidates(
                    states=all_states,
                    embeddings=embeddings,
                    labels=all_labels,
                    min_members=min_discovery_members,
                    min_evr=min_discovery_evr,
                    max_known_corr=discovery_max_corr,
                )
                discovery = DiscoveryReport(
                    coordinate_candidates=coords,
                    gap_candidates=gaps,
                    term_proposals=[],
                    excluded_coordinates=excluded,
                    diagnostics=diagnostics,
                )

            report = build_intelligence_benchmark_report(
                grouped_states=grouped_states,
                language_codes=language_codes,
                embeddings=embeddings,
                labels=all_labels,
                discovery=discovery,
            )

            st.session_state["intel_benchmark_result"] = (
                grouped_states, language_codes, all_states, all_labels,
                embeddings, emodel, report
            )

    if "intel_benchmark_result" in st.session_state:
        grouped_states, language_codes, all_states, all_labels, embeddings, emodel, report = st.session_state["intel_benchmark_result"]

        m = st.columns(8)
        m[0].metric("Languages",report.total_languages)
        m[1].metric("Utterances",report.total_utterances)
        m[2].metric("Tokens",report.total_tokens)
        m[3].metric("Failures",len(report.findings))
        m[4].metric("Lexical",f"{report.benchmark_dimensions.get('lexical_intelligence',0):.2f}")
        m[5].metric("Structural",f"{report.benchmark_dimensions.get('structural_intelligence',0):.2f}")
        m[6].metric("Reference",f"{report.benchmark_dimensions.get('referential_intelligence',0):.2f}")
        m[7].metric("Comprehension",f"{report.benchmark_dimensions.get('comprehension_intelligence',0):.2f}")

        tabs = st.tabs([
            "Intelligence Report",
            "Benchmark Dashboard",
            "Language Profiles",
            "Native Language Maps",
            "Utterance Inspector",
            "Meaning Frames",
            "Lexical Coverage",
            "Failure Ledger",
            "Cross-Language",
            "Discovery",
            "Export",
        ])

        with tabs[0]:
            md = intelligence_report_markdown(
                report,grouped_states,emodel
            )
            st.markdown(md)
            st.download_button(
                "Download intelligence report",
                md.encode("utf-8"),
                "atlas_intelligence_report.md",
                "text/markdown",
                use_container_width=True
            )

        with tabs[1]:
            st.markdown("### Benchmark dimensions")
            dim_df = benchmark_dimension_df(report)
            st.dataframe(dim_df,hide_index=True,use_container_width=True)
            if not dim_df.empty:
                fig = go.Figure(go.Bar(
                    x=dim_df["dimension"],y=dim_df["score"]
                ))
                fig.update_layout(
                    template="plotly_dark",
                    yaxis=dict(range=[0,1]),
                    height=430,
                    title="ATLAS intelligence benchmark dimensions"
                )
                st.plotly_chart(fig,use_container_width=True)

            st.markdown("### Global typed-space coverage / activation")
            cov_df = pd.DataFrame([
                {"space":sp,"coverage_activation":v}
                for sp,v in report.typed_space_coverage.items()
            ]).sort_values("coverage_activation",ascending=False)
            st.dataframe(cov_df,hide_index=True,use_container_width=True,height=520)

        with tabs[2]:
            pdf = intelligence_profile_df(report)
            st.dataframe(pdf,hide_index=True,use_container_width=True,height=520)

            metric_choice = st.selectbox(
                "Compare language profiles by",
                [
                    "lexical_coverage","frame_coverage","role_coverage",
                    "reference_resolution","coreference_resolution",
                    "comprehension","validation","uncertainty"
                ],
                key="intel_profile_metric"
            )
            if not pdf.empty:
                fig = go.Figure(go.Bar(
                    x=pdf["language"],y=pdf[metric_choice]
                ))
                fig.update_layout(
                    template="plotly_dark",
                    height=420,
                    title=f"Language profile — {metric_choice}"
                )
                st.plotly_chart(fig,use_container_width=True)

        with tabs[3]:
            map_language=st.selectbox(
                "Inspect active map",
                list(grouped_states.keys()),
                key="intel_native_map_language"
            )
            map_code=language_codes[map_language]
            bm=language_map_profile(map_code,map_language)
            st.plotly_chart(native_language_map_graph(bm),use_container_width=True)
            st.json({
                "analyzerStatus":bm.analyzerStatus,
                "nativeMechanics":bm.nativeMechanics,
                "operators":bm.nativeOperators,
                "constraints":bm.nativeConstraints,
                "alignmentInterfaces":bm.alignmentInterfaces,
                "observationCoverage":bm.observationCoverage,
            })

        with tabs[4]:
            language_choice = st.selectbox(
                "Language",
                list(grouped_states.keys()),
                key="intel_inspect_lang"
            )
            state_labels = [
                f"{i+1}. {s.form}"
                for i,s in enumerate(grouped_states[language_choice])
            ]
            utterance_choice = st.selectbox(
                "Utterance",
                state_labels,
                key="intel_inspect_utt"
            )
            si = state_labels.index(utterance_choice)
            s = grouped_states[language_choice][si]

            st.markdown("#### Meaning frames")
            st.json([asdict(x) for x in s.meaningFrames])
            st.markdown("#### Reference / coreference")
            st.json({
                "references":[asdict(x) for x in s.references],
                "coreference":[asdict(x) for x in s.coreference],
            })
            st.markdown("#### Semantic roles")
            st.json([asdict(x) for x in s.semanticRoles])
            st.markdown("#### Comprehension certificate")
            st.json(asdict(s.comprehensionCertificate) if s.comprehensionCertificate else {})
            st.markdown("#### State Δ")
            st.json(asdict(s.stateDelta) if s.stateDelta else {})
            st.markdown("#### Counterfactual neighbors")
            st.json([asdict(x) for x in s.counterfactualNeighbors])

        with tabs[5]:
            rows = []
            for lang_name,states in grouped_states.items():
                for s in states:
                    for f in s.meaningFrames:
                        rows.append({
                            "language":lang_name,
                            "utterance":s.form,
                            "frame_type":f.frame_type,
                            "predicate":f.predicate,
                            "canonical":f.canonical_form,
                            "direction":f.direction,
                            "confidence":f.confidence,
                        })
            fdf = pd.DataFrame(rows)
            if fdf.empty:
                st.info("No meaning frames were constructed.")
            else:
                st.dataframe(fdf,hide_index=True,use_container_width=True,height=620)

        with tabs[6]:
            lexical_rows = []
            for lang_name,states in grouped_states.items():
                for s in states:
                    cov = lexical_coverage_df(s)
                    if not cov.empty:
                        cov = cov.copy()
                        cov.insert(0,"utterance",s.form)
                        cov.insert(0,"language",lang_name)
                        lexical_rows.append(cov)
            if lexical_rows:
                st.dataframe(
                    pd.concat(lexical_rows,ignore_index=True),
                    hide_index=True,use_container_width=True,height=620
                )
            else:
                st.info("No lexical coverage rows.")

        with tabs[7]:
            fdf = intelligence_findings_df(report)
            if fdf.empty:
                st.success("No benchmark failures were emitted.")
            else:
                severity_filter = st.multiselect(
                    "Severity",
                    ["HIGH","MEDIUM","LOW"],
                    default=["HIGH","MEDIUM","LOW"],
                    key="intel_failure_severity"
                )
                st.dataframe(
                    fdf[fdf["severity"].isin(severity_filter)],
                    hide_index=True,use_container_width=True,height=650
                )

        with tabs[8]:
            if report.cross_language_alignment:
                st.json(asdict(report.cross_language_alignment))
            if report.invariant_candidate:
                st.json(asdict(report.invariant_candidate))
            if len(all_states) >= 2:
                st.plotly_chart(
                    embedding_plot(embeddings,all_labels),
                    use_container_width=True
                )

        with tabs[9]:
            if report.discovery is None:
                st.info("Discovery was not run or the benchmark was too small.")
            else:
                st.markdown("#### Candidate coordinates")
                st.dataframe(
                    discovery_coordinate_df(report.discovery),
                    hide_index=True,use_container_width=True
                )
                st.markdown("#### Persistent gaps")
                st.dataframe(
                    discovery_gap_df(report.discovery),
                    hide_index=True,use_container_width=True
                )
                with st.expander("Discovery diagnostics"):
                    st.json(report.discovery.diagnostics)

        with tabs[10]:
            md = intelligence_report_markdown(
                report,grouped_states,emodel
            )
            machine = {
                "report":asdict(report),
                "states":{
                    lang:[asdict(s) for s in states]
                    for lang,states in grouped_states.items()
                },
                "embedding_model":emodel,
            }
            st.download_button(
                "Download intelligence report Markdown",
                md.encode("utf-8"),
                "atlas_intelligence_report.md",
                "text/markdown",
                use_container_width=True
            )
            st.download_button(
                "Download intelligence report JSON",
                json.dumps(machine,ensure_ascii=False,indent=2).encode("utf-8"),
                "atlas_intelligence_report.json",
                "application/json",
                use_container_width=True
            )
            st.download_button(
                "Download language profiles CSV",
                intelligence_profile_df(report).to_csv(index=False).encode("utf-8"),
                "atlas_language_profiles.csv",
                "text/csv",
                use_container_width=True
            )
            st.download_button(
                "Download failure ledger CSV",
                intelligence_findings_df(report).to_csv(index=False).encode("utf-8"),
                "atlas_failure_ledger.csv",
                "text/csv",
                use_container_width=True
            )


elif mode == "Active Native Language Maps":
    st.subheader("ATLAS Active Native Language Map Explorer")
    lang_name=st.selectbox("Language map",list(LANGUAGES.keys()),key="active_map_explorer")
    lang=LANGUAGES[lang_name]
    lmap=language_map_profile(lang,lang_name)
    analyzer=get_active_language_map(lang)

    m=st.columns(5)
    m[0].metric("Language",lang_name)
    m[1].metric("Analyzer",lmap.analyzerStatus.get("class",""))
    m[2].metric("Native spaces",sum(1 for v in lmap.typed_spaces.values() if v))
    m[3].metric("Native axes",sum(len(v) for v in lmap.typed_spaces.values()))
    m[4].metric("Operators",len(lmap.nativeOperators))

    tabs=st.tabs([
        "Map Geometry","Native Spaces","Mechanics","Operators","Constraints",
        "Alignment Interfaces","Analyzer Probe","Raw State"
    ])

    with tabs[0]:
        st.plotly_chart(native_language_map_graph(lmap),use_container_width=True)

    with tabs[1]:
        rows=[]
        for sp,coords in lmap.typed_spaces.items():
            for c,v in coords.items():
                rows.append({"space":sp,"coordinate":c,"value":v,"description":SPACE_DESCRIPTIONS.get(sp,"")})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=650)

    with tabs[2]:
        st.json(lmap.nativeMechanics)

    with tabs[3]:
        st.dataframe(pd.DataFrame(lmap.nativeOperators),hide_index=True,use_container_width=True)

    with tabs[4]:
        st.dataframe(pd.DataFrame(lmap.nativeConstraints),hide_index=True,use_container_width=True)

    with tabs[5]:
        st.json({
            "metrics":lmap.nativeMetrics,
            "alignmentInterfaces":lmap.alignmentInterfaces,
            "contextFibers":lmap.contextFibers,
        })

    with tabs[6]:
        probe=st.text_area("Probe utterance","I love myself." if lang=="en" else "",height=110,key="native_probe_text")
        if st.button("Run through this native map",use_container_width=True,key="native_probe_run"):
            if probe.strip():
                s=make_utterance_state(probe.strip(),lang)
                st.markdown("#### Native mechanics observed")
                st.json(s.nativeMechanicsObserved)
                st.markdown("#### Tokens")
                st.dataframe(pd.DataFrame([asdict(ts.token) for ts in s.tokens]),hide_index=True,use_container_width=True)
                st.markdown("#### Meaning frames")
                st.json([asdict(f) for f in s.meaningFrames])
                st.markdown("#### Roles / reference")
                st.json({
                    "references":[asdict(r) for r in s.references],
                    "coreference":[asdict(c) for c in s.coreference],
                    "roles":[asdict(r) for r in s.semanticRoles],
                })

    with tabs[7]:
        st.json(asdict(lmap),expanded=False)


elif mode == "Language Map ↔ Language Map":
    st.subheader("Active Native Language Map Comparison")
    st.caption(
        "Maps are compared only on explicitly shared native coordinates. "
        "A coordinate absent from either map is NON-COMPARABLE, not zero."
    )
    c1,c2=st.columns(2)
    with c1:
        la_name=st.selectbox("Language map A",list(LANGUAGES.keys()),key="lm_a")
    with c2:
        lb_name=st.selectbox("Language map B",list(LANGUAGES.keys()),index=1,key="lm_b")

    if st.button("Compare active language maps",type="primary",use_container_width=True):
        la,lb=LANGUAGES[la_name],LANGUAGES[lb_name]
        lma,lmb=language_map_profile(la,la_name),language_map_profile(lb,lb_name)
        summary_df,coord_df=compare_language_maps_native(lma,lmb)
        st.session_state["lm_result"]=(lma,lmb,summary_df,coord_df)

    if "lm_result" in st.session_state:
        lma,lmb,summary_df,coord_df=st.session_state["lm_result"]

        comparable=int((summary_df["shared_coordinate_count"]>0).sum())
        m=st.columns(5)
        m[0].metric("Map A analyzer",lma.analyzerStatus.get("class",""))
        m[1].metric("Map B analyzer",lmb.analyzerStatus.get("class",""))
        m[2].metric("Comparable spaces",comparable)
        m[3].metric("A native axes",sum(len(v) for v in lma.typed_spaces.values()))
        m[4].metric("B native axes",sum(len(v) for v in lmb.typed_spaces.values()))

        tabs=st.tabs([
            "Shared Native Geometry","Coordinate Alignment",
            "Map A","Map B","Map Graphs","Operators / Constraints","Alignment Interfaces"
        ])

        with tabs[0]:
            plot_df=summary_df.dropna(subset=["native_axis_coherence"])
            if plot_df.empty:
                st.info("These maps currently have no explicitly shared native axes.")
            else:
                fig=go.Figure(go.Bar(x=plot_df["space"],y=plot_df["native_axis_coherence"]))
                fig.update_layout(template="plotly_dark",height=430,yaxis=dict(range=[0,1]),
                                  title="Coherence over shared native axes only")
                st.plotly_chart(fig,use_container_width=True)
            st.dataframe(summary_df,hide_index=True,use_container_width=True,height=520)

        with tabs[1]:
            if coord_df.empty:
                st.info("No coordinate-level native alignment available.")
            else:
                st.dataframe(coord_df,hide_index=True,use_container_width=True,height=620)

        with tabs[2]:
            st.plotly_chart(native_language_map_graph(lma),use_container_width=True)
            st.json(asdict(lma),expanded=False)

        with tabs[3]:
            st.plotly_chart(native_language_map_graph(lmb),use_container_width=True)
            st.json(asdict(lmb),expanded=False)

        with tabs[4]:
            ca,cb=st.columns(2)
            with ca:
                st.plotly_chart(native_language_map_graph(lma),use_container_width=True,key="map_graph_a")
            with cb:
                st.plotly_chart(native_language_map_graph(lmb),use_container_width=True,key="map_graph_b")

        with tabs[5]:
            st.json({
                lma.language_name:{
                    "operators":lma.nativeOperators,
                    "constraints":lma.nativeConstraints,
                    "mechanics":lma.nativeMechanics,
                },
                lmb.language_name:{
                    "operators":lmb.nativeOperators,
                    "constraints":lmb.nativeConstraints,
                    "mechanics":lmb.nativeMechanics,
                }
            })

        with tabs[6]:
            st.json({
                lma.language_name:lma.alignmentInterfaces,
                lmb.language_name:lmb.alignmentInterfaces,
            })


st.divider()
st.caption(
    "ATLAS v10 active-native-map prototype: language maps are computational analyzers rather than "
    "generic complexity profiles. English, Persian, Arabic, Mandarin, and Turkish use dedicated native "
    "analysis paths; other registered languages expose explicit structural profiles until dedicated "
    "analyzers are implemented. Missing native coordinates are non-observations, not zero-valued geometry."
)
