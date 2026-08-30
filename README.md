Install:

pip install -r requirements.txt

streamlit run ATLAS.py

------------------------------------------

Current version:

ATLAS v10 — Active Native Language Maps

------------------------------------------

Dedicated analyzers currently include:

English

Persian / Farsi

Arabic

Mandarin Chinese

Turkish

------------------------------------------

Features:

Word and utterance analysis

Typed language states

Semantic roles

Reference and coreference

Meaning frames

Lexical relations

Synonyms / antonyms / related terms

Active native language maps

Cross-language comparison

Invariant and residual detection

Counterfactual transformations

Intelligence benchmark

Failure / falsification ledger

Candidate coordinate and gap discovery

Interactive Streamlit lab

-----------------------------------------

This code implements an interactive research laboratory (Streamlit app) called ATLAS Multi-Space Training Data Foundry. It is designed to transform natural language utterances into richly typed, multi‑dimensional linguistic states, compare them across languages, discover latent structure, and generate versioned training corpora for downstream supervised learning. The system is built around a core epistemic principle: unknown is not zero; similarity is not a typed relation; display projections are not native state.

1. Overall Architecture
The code is organized into several layers:

Data Models – Python dataclasses representing the ATLAS object hierarchy (utterance, token, word, language map, top layer) and all supporting structures (relations, hyperrelations, transformations, uncertainty, validation, candidates, etc.).

Core Utilities – Tokenization, script detection, cue matching, POS guessing, lexical relation extraction, uncertainty/validation heuristics.

Active Language Maps – Language‑specific analyzers that tokenize, assign POS, morphology, reference, semantic roles, and meaning frames according to native grammatical mechanics.

Embedding Layer – Computes sentence embeddings (via SentenceTransformer or hashing) and optionally trains a shallow autoencoder for latent representation.

Comparison & Alignment Engine – Compares two or more utterance states, computes typed‑space deltas, extracts invariants, residuals, and cross‑language alignments.

Discovery Engine – Uses PCA on embeddings to find unexplained structure; proposes new coordinates, persistent gaps, and lexical term requirements.

Intelligence Benchmark – Runs a corpus of utterances across multiple languages, produces per‑language profiles, failure findings, and global metrics.

Dataset Aggregation – Normalises all generated states into objects, coordinate facts, edge facts, metadata facts, and training examples (with eligibility gates). The data is stored in an SQLite spine and can be exported as JSONL/CSV/ZIP.

Streamlit UI – Provides a sidebar for experiment selection, model configuration, persistence settings, and 10 distinct experiment modes, each with its own input panels and result visualisations.

2. Core Data Structures
2.1 Typed Spaces
There are 34 typed spaces (e.g., ORTH, PHON, LEX, POS, SEM, LOG, AFFECT, UNC, COMP). Each space is a dictionary of named coordinates, each holding a float between 0 and 1. Coordinates are heuristic or provisional, derived from surface cues, lexical resources, and native map specifications. The COMP space now contains derived comprehension scores (structural coverage and semantic resolution) rather than a naïve token‑integration average.

2.2 Object Hierarchy
Token – surface form, lemma, POS, morphology, span, language.

TokenState – wraps a Token with its typed spaces, relations, hyperrelations, uncertainty, validation, and lexical neighbourhood.

WordState – a lexical carrier above the tokeniser (currently one orthographic token per word, but allows future multi‑token carriers).

UtteranceState – aggregates token states, adds utterance‑level typed spaces, relations, hyperrelations, transformations, reference entities, coreference, semantic roles, meaning frames, comprehension certificate, state delta, and counterfactual neighbours.

LanguageMapState – describes a language’s native mechanics, operators, constraints, typed spaces, and alignment interfaces. It is produced by an active language‑map analyzer.

TopLayerState – collects all language maps, utterances, alignments, invariants, residuals, coordinate candidates, gaps, and term proposals.

2.3 Lexical Relations
A LexicalNeighborhood is built per token using WordNet (if available) or a curated fallback. It contains lexical senses and typed relations (synonym, antonym, hypernym, meronym, etc.) with confidence weights. Missing relation types are recorded as unobserved, not zero.

2.4 Comprehension Certificate
A structural certificate that tracks which components of an utterance are resolved (predicate, subject, object, coreference, affect target, scope). It produces a comprehension_score based on structural coverage and semantic resolution, with explicit falsification tests.

3. Active Language Maps
The code implements a plugin‑like system for language‑specific processing. A base AtlasLanguageMap class provides generic tokenisation, POS guessing (using cue matching and a small curated lexicon), and minimal role/frame extraction. For a subset of languages (English, Persian, Arabic, Mandarin, Turkish), there are dedicated subclasses that override tokenisation, POS/morphology, reference resolution, semantic‑role assignment, and meaning‑frame construction according to native grammar.

Each map exposes:

tokenize() – splits text into tokens using language‑aware rules (e.g., maximal‑match for Mandarin).

analyze_pos() – assigns POS using native lexical lists and heuristics.

resolve_reference() – identifies speaker, addressee, and reflexive pronouns.

assign_roles_and_frames() – builds semantic role bindings and meaning frames (affective, epistemic, propositional) using language‑specific patterns (e.g., Persian light verbs, Turkish case/agreement).

native_mechanics_observed() – records which operators (e.g., negation, evidentiality) are active.

The map is cached via @st.cache_resource so it is built once per language.

4. Processing Pipeline for an Utterance
When a user enters text and selects a language, the following steps occur:

Tokenisation – The active language map tokenises the text into Token objects.

Token‑level typed spaces – For each token, the map calls encode_native_token_state(), which fills the 34 typed spaces with heuristic values (e.g., script features, POS one‑hots, cue‑based flags for negation, modality, affect). Native map axes are added under keys like native::....

Lexical Neighbourhood – For each token, build_lexical_neighborhood() builds dictionary‑backed senses and relations (WordNet or fallback).

Reference and Role Resolution – The map resolves reference entities (speaker, addressee), coreference chains, and semantic roles (agent, experiencer, theme, target, etc.).

Meaning‑Frame Construction – Based on predicates and their arguments, frames are constructed for affective attitudes (love/hate), epistemic attitudes (believe/know), and general predicate‑argument frames. For the metalinguistic predicate mean, a special frame is created.

Aggregation – Token states are aggregated to produce utterance‑level typed spaces (using max for certain boolean‑like coordinates, mean for the rest).

Relations and Hyperrelations – The map extracts surface‑adjacency relations and more structured relations (negation scope, modal scope, contrast, causation, etc.) from the token stream. Hyperrelations combine multiple nodes into frames (e.g., epistemic proposition frame).

Comprehension Certificate – A certificate is built that checks whether predicate, subject, object, coreference, affect target, and scope are resolved, producing a structural coverage score, semantic resolution, and a comprehension score.

State Delta – A delta object estimates update potentials for knowledge, belief, affect, self, social relation, and pragmatics.

Counterfactual Neighbours – For English, generate variants by flipping affect polarity, changing target (self/other), or adding negation.

Embedding – The input texts are embedded (SentenceTransformer or hashing). Optionally, an NNS autoencoder is trained to produce a latent representation.

The result is an UtteranceState object containing all the above.

5. Comparison and Alignment
5.1 Pairwise Comparison (compare_two_states)
Given two utterance states and their embeddings, it computes:

Embedding cosine similarity.

Typed‑space deltas (mean activation differences per space).

Candidate invariant spaces (absolute delta ≤ 0.08).

Meaning‑frame comparison (frame type, predicate, direction, affect valence, role skeleton).

5.2 Many‑Member Alignment (align_many)
For two or more states, it:

Computes an embedding centroid and coherence.

Flattens all observed coordinates (ignoring unknowns and constants) into a feature matrix.

For each coordinate, calculates a group mean and standard deviation; coordinates with low coverage or constant values are excluded from invariant evidence.

Produces an Alignment object with embedding and per‑space coherence.

Produces an InvariantCandidate with confidence based on the coherence of non‑constant coordinates, embedding coherence, and coverage.

Produces a residual DataFrame showing per‑member, per‑coordinate deviation from the group mean.

6. Discovery Engine (discover_coordinate_candidates)
This engine tries to find new structural dimensions that are not explained by the current set of observed ATLAS coordinates.

Feature Matrix – All observed coordinates (coverage ≥ 60%) are standardised (z‑score). Coordinates that are constant or have very low variance are excluded.

PCA on Embeddings – A PCA is run on the utterance embeddings (the observation layer). Each principal component (PC) is considered a potential new coordinate.

Correlation Check – For each PC, the absolute correlation with every existing standardised coordinate is computed. If the maximum correlation exceeds a threshold (default 0.70), the PC is considered explainable and rejected.

Novelty Score – A heuristic combining explained variance, independence from known coordinates, and number of members.

Gap Candidates – If a PC passes the gate, a GapCandidate is created with a recommended test and competing explanations.

Term Proposals – Using declared multilingual entries (with optional ? placeholders), the engine proposes new lexical terms for languages that are missing a lexicalisation for the concept, especially when an invariant candidate exists.

The output is a DiscoveryReport with coordinate candidates, gap candidates, and term proposals, plus diagnostics.

7. Intelligence Benchmark
The Language Intelligence Benchmark mode runs a corpus of utterances across multiple languages. It:

Parses the input (either [Language] sections or Language | utterance lines).

Constructs an UtteranceState for each utterance using the respective active language map.

Builds a LanguageIntelligenceProfile per language, aggregating lexical coverage, frame coverage, role coverage, reference/coreference resolution, comprehension confidence, validation confidence, uncertainty, and typed‑space coverage.

Accumulates a global failure ledger (BenchmarkFinding) for unresolved POS, missing lexical resources, unresolved predicate/subject/object, low validation confidence, high uncertainty, and missing frames.

Computes benchmark‑dimension scores (lexical, structural, referential, comprehension, validation, uncertainty control, cross‑language balance).

Optionally runs the discovery engine across the entire member set.

Produces a comprehensive IntelligenceBenchmarkReport with profiles, inventories, findings, and cross‑language alignment/invariant.

8. Dataset Aggregation and Persistence
8.1 build_atlas_dataset_run
After any experiment, this function normalises all generated states, language maps, comparisons, alignments, invariants, residuals, discovery outputs, and benchmark reports into a structured dataset run. It creates:

Objects – one row per carrier (language map, utterance, word, token, meaning frame, etc.) with identifiers, types, and metadata.

Coordinate Facts – long‑form rows for each typed coordinate with its value, evidence class, observation status, source, method, and a flag indicating whether it is unknown.

Edge Facts – relations, hyperrelations, coreference edges, and semantic‑role bindings.

Metadata Facts – provenance, validation, uncertainty, attention, comprehension, knowledge delta, native mechanics, etc.

Training Examples – each row is a supervision candidate with task_type, input, target, evidence, confidence, observation status, eligibility flag, group fingerprint, deterministic split (80/10/10), and an is_negative flag. Examples are generated for all contracts defined per experiment mode.

8.2 Persistence
The AtlasSQLiteStore class writes all these tables into a SQLite database. It is append‑only and stores schema versions, content‑addressed sources, source records, lineage edges, and recursive data products. The store is enabled via the sidebar toggle.

8.3 Export
The observatory tab provides:

Download of the complete bundle as JSON or ZIP (including CSV tables and sharded JSONL per task/split).

Import of prior bundles (merging runs).

Materialisation of aggregated data products (e.g., language‑space coverage matrix, relation inventory, drift table) as versioned products with lineage.

9. Experiment Modes
The UI offers 10 modes, each with a specific training contract. The code routes each mode to a dedicated panel that collects inputs, runs the appropriate pipeline, stores results, and displays them via a set of tabbed views. Common output tabs include:

Clean Report – human‑readable markdown summary of all states, frames, roles, certificates, residuals, discovery, and language maps.

Meaning Frame – visualisation of resolved frames.

Reference + Roles – entities, coreference, semantic roles, state delta.

Counterfactuals – generated structural variants.

Top Layer / Language Map / Utterance State – raw JSON views.

Each Word / Token – detailed inspection per token.

Lexical Relations – graph and tables of senses/edges.

Fibers – 3D plot of token‑wise typed‑space activations.

Uncertainty / Validation – numeric summaries.

Embedding / NNS – PCA projections and autoencoder losses.

The Data / Metadata Observatory mode is a central dashboard that shows all collected runs, tables, evidence coverage, drift, training material, source catalog, and data products.

10. Key Epistemic Rules Enforced in Code
Unknown ≠ zero – coordinates marked _unknown or with low coverage are not treated as zero; they are excluded from invariance and residual calculations.

Similarity ≠ relation – embedding similarity is never used as evidence for a typed relation; relations come from explicit structural heuristics or lexical resources.

Shared default ≠ invariant – coordinates that are constant across all members are rejected as invariant evidence because they may be encoder defaults.

Display projection ≠ native state – PCA or t‑SNE projections are only visual aids; typed coordinates are the source of truth.

Candidate ≠ fact – coordinate candidates, gap candidates, and term proposals are flagged as CANDIDATE or PROVISIONAL until independently validated.
