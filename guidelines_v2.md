# PII annotation guidelines v2 — canonical

Ratified 2026-07-18 (Harsh + adjudication audit). These rules govern: the
corrected eval gold, v2 training-data relabeling, prompt-embedded guideline
conditioning, and all future adjudication rubrics. Derived from the error
autopsy and the Job-A adjudication of 2,148 gold spans (~9% noise found).

## Date family
- `date` — a calendar date with NO time component (e.g. 2023-08-15).
- `date_time` — must contain BOTH a date and a time (2023-10-01T10:00:00Z).
  A bare date is never date_time. (Mechanically checkable.)
- `time` — a clock time alone.
- `date_of_birth` — a date explicitly denoting a person's birth; takes
  precedence over `date`.
- Date fragments inside identifiers (INV-2024-07-01) are not dates.
- Individual endpoints of a range ("2005-2010") are not tagged separately
  (ruling [10], 2026-07-18).

## Occupation
- The full job title or role of a person, including compound descriptors
  ("laborer freight stock or material mover", "Assistant Coach").
- EXCLUDES: department/organization suffixes ("... department"), document
  titles ("Author Interview Transcript"), honorific prefixes, bare
  credentials (MD, PhD), and generic mentions of a job category not
  attached to a person ("innovations in computer support specialist roles").

## Entityhood (the semantics stance)
- **Attribute semantics**: an entity is a span disclosing information about
  a person or their record. World-fact mentions of countries, states, and
  organizations in encyclopedic/descriptive context are NOT entities
  (ruling [3], 2026-07-18). A mention-semantics reading may later be made
  available as an explicitly conditioned request style; it is not the
  default.
- **Third-party PII counts**: attributes of ANY natural person in the
  document are entities, not only the document's primary subject
  (ruling [9], 2026-07-18).
- Field labels are never values ("Temporary Password", "Policyholder Last
  Name" as literal form-field names).
- Empty placeholders, blank form fields ("State of ________"), checkbox
  glyphs, and template scaffolding are never entities.
- No entity is shorter than 2 characters (enforced in the decoder).
