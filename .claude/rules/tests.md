---
paths:
  - "tests/**"
---

# Test names

A test name states the claim the test asserts — what is true, not how it is checked or how it
came to be. The frame is chapter 1 of Zachary Tellman's *Elements of Clojure*: a name is
**narrow** when it excludes every test it is not, and **consistent** when one sense always gets
one sign. A reader who sees two different verbs will assume two different behaviours.

## One verb per kind of refusal

| Verb | Means | Example |
|---|---|---|
| `rejects` | an input refused at a boundary — HTTP 422, pandera validation | `test_predict_rejects_unknown_field` |
| `refuses` | the system declines to start, serve, export or promote because of configuration or an artifact | `test_the_lifespan_refuses_a_track_whose_model_cannot_score_its_example` |
| `raises` | only when the specific exception type is the contract a caller catches | `test_competition_403_raises_a_consent_error_naming_the_rules_url` |

The status code belongs in the assert, not the name: `rejects` already fixes that the refusal
happened at the request boundary. Voice is free — `is_rejected` and `is_refused` carry the same
sense as the active verb, and paired tests (`…is_accepted` / `…is_refused`) read better matched.
What must not vary is the verb. The same holds for acceptance: `accepts`, not `permits` or
`tolerates`.

## Signs that are banned

- **`returns`** — every function returns something, so the word adds no sense; the meaning is
  carried entirely by its object, and `returns_422` gave three different failures the same
  name. Name what the returned value *says*.
- **`now`, `still`, `actually`, `would_have`** — they only carry information on the day the
  test was written, relative to a history the reader does not have. State the present claim.
- **`happy_path`, bare nouns (`test_health`, `test_band_collapse`)** — they fit any test of
  that unit, so they exclude nothing. Say what the happy path produces.
- **A reason clause (`which_is_why_…`)** — the name holds the claim; the reason goes in the
  docstring.

## Same situation, same words across tracks

`tests/test_ingest.py` and `tests/test_fraud.py` test the same machinery against two datasets. A dropped
column is `reports_a_dropped_…` in both, a renamed one `names_a_renamed_…`, and the validator
is `validate_` in both — not `validation_` in one file and `validate_` in the other.

## Renaming

Collect before and after — `uv run pytest --collect-only -q | grep '::' | sort` — and diff the
two lists. The count must not change: a name that loses its `test_` prefix is not a failure, it
is a test that silently stops running. Then grep `STATUS.md`, `docs/` and `.claude/` for the
old names; PR bodies cannot be edited, and `git log -S <old_name>` is how the old name is traced.
