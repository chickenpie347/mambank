# Artificial Memory Prosthesis — Toy Model v0.1

**Purpose:** demonstrate, at software level, how an external episodic memory store might support recall when a person's own retrieval cues are incomplete.

## What is modelled
- Episodic records with time, person, event, place, details, confidence and source.
- Content-addressed deduplication using SHA-256.
- A simulated BCI interface that passes an abstract cue to the external memory.
- Similarity-based retrieval of candidate episodes.
- Two toy impairment controls: missing stored episodes (encoding loss) and degraded cue text (retrieval impairment).
- Explicit human confirmation: retrieved candidates are not treated as facts until confirmed.

## Relationship to the uploaded MemBank paper
The model borrows high-level ideas from *MemBank: Pointer-Based Neural Activation Memory for Open-Source LLMs* (uploaded paper): content-addressed identifiers, external persistent memory, retrieval, and separation between the model adapter and storage. It does **not** implement the paper's activation-tensor pointers, HGNS compression hierarchy, transformer hooks, or neural activation retrieval. The paper describes LLM hidden-state storage; this toy instead stores human-readable episodic records.

## What is deliberately not modelled
- No neural signals, electrodes, neural decoding, stimulation, hippocampal model, or clinical data.
- No Alzheimer's diagnosis, treatment claim, efficacy estimate, or patient-specific recommendation.
- No inference that human memory is equivalent to LLM memory.
- No autonomous write-back of unverified memories.

## Run
Requires Python 3.9+; standard library only.

```bash
python memory_prosthesis_toy.py
```

## Interpretation
The demo illustrates a narrow proposition: when a cue is incomplete, an external store may surface candidate records that a person or trusted caregiver can verify. The encoding-loss switch illustrates a hard boundary: external retrieval cannot recover an event that was never captured or whose record is unavailable.

## Research questions for later versions
1. What information can be captured reliably and consensually in daily life?
2. How should the system represent uncertainty, conflicting accounts and memory corrections?
3. Can a BCI detect a retrieval-intent or cue state reliably enough to initiate assistance?
4. How should the system avoid false familiarity, suggestion effects and caregiver-induced errors?
5. Which cognitive tasks, if any, improve with assistive recall, and how would this be measured safely?

**Status:** conceptual toy model; not validated; not a medical device; not suitable for clinical use.
