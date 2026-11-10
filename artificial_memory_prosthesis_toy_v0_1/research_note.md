# Research Note: Artificial Memory Prosthesis Toy Model v0.1

## Hypothesis
An external, provenance-aware episodic store, accessed through an assistive interface, may help a person recover useful contextual information when biological recall is impaired.

## Minimal architecture
1. **Capture layer:** consensually records selected events and context.
2. **External episodic store:** stores records with timestamps, entities, provenance and confidence.
3. **Cue interface:** receives a user-generated or, in a future research system, BCI-derived retrieval cue.
4. **Retrieval layer:** returns ranked candidate records.
5. **Confirmation layer:** asks the person or an authorised support person to confirm, correct or reject the candidate.
6. **Presentation layer:** returns a gentle, contextual cue rather than asserting that the system knows what the person remembers.

## Toy-model mapping
- Capture = manually authored Episode records.
- External store = in-memory content-addressed dictionary.
- BCI = text pass-through placeholder, not a neural decoder.
- Retrieval = token-set cosine similarity.
- Encoding impairment = restricting which episode IDs are available.
- Retrieval impairment = removing selected words from the cue.
- Confirmation = explicit warning that suggestions require human verification.

## Critical limitation
The toy model is a demonstration of software retrieval, not a model of Alzheimer's pathology. Alzheimer's can affect encoding, consolidation, retrieval, association, orientation, language and other cognitive functions through progressive biological changes. A memory store can only assist with information that has been captured accurately and remains accessible. It cannot recreate an uncaptured experience or restore damaged neural networks by itself.

## Safety and ethics requirements before any human study
- Informed, ongoing consent and capacity-sensitive consent procedures.
- Patient control over capture, review, correction, export and deletion.
- Strict access control, encryption and audit logs.
- Clear separation of patient-entered, caregiver-entered, sensor-derived and system-inferred information.
- No confident presentation of uncertain or conflicting records.
- Testing for false memories, suggestion, over-reliance and distress.
- Clinical, ethics and regulatory review before any patient-facing study.
- BCI acquisition/stimulation must be designed and supervised by qualified clinical and neural-engineering teams.

## Evidence boundary
The uploaded MemBank paper supports the existence of a proposed LLM activation-memory architecture using content-addressed pointers, hierarchical compression and coarse-to-fine retrieval. It does not establish that those mechanisms are suitable for human episodic memory or BCI integration. The present design is an extrapolative research sketch.
