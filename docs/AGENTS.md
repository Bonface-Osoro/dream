# AGENTS.md

Coding rules for this repository. Follow every time without exception.
Adapted from the rules used in spacewx_cyber.

## Repository scope

hesabu ingests scanned IEBC polling station declaration forms (Form 34A for
the presidential race, Form 35A for the other races), reads the printed header
and the handwritten counts with a vision language model served by Ollama on
this machine, validates the numbers, and accumulates one row per polling
station. Images never leave this machine.

The Python package lives in `src/hesabu`. The web page lives in `static/`.
The 25 Form 35A samples used for calibration live in `samples/`. The archive
of Form 34A scans lives in `data/raw/`. Working state (uploads, records, debug
crops, results) lives in `storage/`. The assessment and the order of work live
in `docs/improvement_plan.md`. Read it before changing extraction code.

## Safety

- Never commit code. Leave completed work in the working tree for the owner.
- Never add `data/raw/`, `raw.zip`, `.venv/`, or credentials to Git.
- Never send a form image, or any crop of one, to a service outside this
  machine. The design decision is local inference only. This rule changes
  only by the owner's written decision in `docs/improvement_plan.md`
  section 0, and this line changes with it.
- Do not change a prompt, a crop geometry, an image scaling constant, or a
  validation rule without running the evaluation set before and after and
  recording both numbers in the report.
- Do not edit files under `samples/`. They are the fixed evaluation set.
- Never add a form image or a crop of one to Git, under any directory.
  `samples/` and `storage/debug_crops/` are already tracked and their
  future is an owner decision recorded in `docs/improvement_plan.md`
  section 0. Do not add to them.
- Do not send a form image to any hosted service, and do not label forms,
  until the image policy in `docs/improvement_plan.md` section 0 is settled.

## Module headers

Every Python module starts with:

```
"""
Role: <one line, what this module is>
Description: <a few lines, what it does and important details>
Author: Bor
"""
```

Do not change the Author field.

## Prose style

No em dashes anywhere. Not in code, comments, docstrings, log messages,
commit messages, or Markdown. Use a comma, a colon, or two sentences.

No semicolons in prose. Code semicolons follow Python rules.

Short scientific sentences. No preamble. Forbidden filler:
"now", "let's", "we'll", "here we", "essentially", "basically",
"simply", "just". If a comment says "now do X", delete "now".

## Comments

Comments explain what is not obvious from the code. They never restate
the code in English.

No section dividers like `# ---- Auth ----` or `# ===== Fetch =====`.
Function names and module structure are the dividers.

No step narration like `# 1. authenticate`, `# 2. fetch`. The log lines
already announce these steps at runtime.

If a comment exists, it earns its space.

## Logging

Use the standard library: `import logging` then
`log = logging.getLogger(__name__)`. A shared `hesabu.logger` module may
replace this later. Switch every module when it exists.

Never use `print()` in library code. Logger only. A script that serves as an
entry point may print its own result summary.

Format with `%s` placeholders rather than f-strings, so the logger can format
lazily and so structured backends for logs keep their fields.

`log.info` for milestones, `log.warning` for recoverable issues,
`log.error` for failures.

## Naming

Functions describe what they do, not how. `read_station_counts` not
`do_vlm_call`. `decode_station_qr` not `process_image`.

Private helpers prefix with `_`.

Constants in module scope are UPPER_SNAKE_CASE.

## Imports

Standard library, then third party, then this repository. Blank line
between groups. One import per line. No `from x import *`.

The local package is `hesabu`. Import siblings with relative imports inside
the package, `from . import results_store`, so the package runs the same way
installed and in the editable environment.

## Type hints

Modern syntax: `list[int]`, `dict[str, float]`, `int | None`.

Type all public function signatures. Internal helpers may skip when
types are obvious from a body of one line.

## Errors

Never silently swallow errors. Catch what you can act on, log the rest,
re-raise when in doubt.

Library code does not exit the process. Only a script that serves as an entry
point may call `sys.exit`.

## Tests

Tests live in `tests/`, mirror the source layout. Plain `pytest`.

Test names describe behaviour, not the function:
`test_station_code_with_fourteen_digits_is_rejected` rather than
`test_validate`.

Tests that need a model call are marked and skipped by default. The unit
suite runs without Ollama.

## Things never to do

- Emoji in code, comments, or commit messages.
- ASCII art banners.
- Unicode smart quotes or fancy bullets in code or docs.
- Placeholder TODOs without an attached issue number.
- Put example values in a model prompt that the model could echo back as
  data. The first row of `storage/results.csv` is a prompt example that came
  back as a station count. Describe the shape, give no numbers and no names.
- Write a station row whose station code fails the 15 digit format, or
  disagrees with the QR code on the form when the QR code decoded.
- Trust a single model read of a number. Every count written to results has
  passed the checks in `results_store`, and the row carries its flags.
- Add a candidate column from a model read of a name. Candidate names come
  from the roster for the election. A name that does not match the roster is
  a flag, not a column.
- Use a hosted model API for any form image.

# ! Critical

## Defaults and missing data

A silent default is a lie. If a value cannot be retrieved or computed, the
right behaviour is one of these, in this preference order:

1. Return `None` or `pd.NA` and let the caller decide.
2. Raise an explicit exception with a message naming the missing input.
3. Log a WARNING and return a sentinel only when the caller has documented
   that it accepts and handles the sentinel.

Never do any of these:

- Substitute a "reasonable" value, for example a vote count of 0, when the
  real value could not be read. A zero that was never on the form changes a
  tally and looks like a measurement.
- Use `dict.get(key, fallback)` for a count or a code. Use `dict.get(key)`
  and let `None` propagate, or check membership and raise.
- Wrap a computation in `try/except Exception: return default_value`.
  Catch only the specific error class you expect, and only when the
  recovery path is documented.
- Fill NaN columns with zero before a sum. A national tally built on
  `fillna(0)` hides every unread station inside the total. Sum with
  `skipna=True` and report the count of unread stations next to the total.

There is no quiet default mode in this repository. A count that could not be
read is `None`, the row is flagged, and the flag names the field.

When you see a default that fails this test, raise an issue, do not fix
silently. The fix changes behaviour and needs an explicit review.

## Agent reporting

A report is an engineering record, not an announcement. Another agent must be
able to act on it without rerunning the work.

State what was verified and how. A claim without a check behind it is a guess,
so label it as one. Distinguish "the suite passed" from "I read the code and it
looks correct".

State what was not done. Skipped tests, unrun evaluations, and model pulls
that were avoided all belong in the report. Silence reads as completion.

Report failures with the evidence attached. Quote the error, name the file and
the line, and give the command that reproduces it. A failure described only in
prose costs the next agent an hour.

Freeze means the change is complete in the working tree and uncommitted. Say so
explicitly, because the reader cannot see the index.

Numbers carry units and denominators. "Accuracy 0.92" is incomplete. "23 of 25
station totals matched the labelled value" is not.

Do not assess the quality of your own work. Report the measurement and let the
reviewer judge. Do not describe what you built as robust, comprehensive, or
ready for production.

When a reviewer files a finding, answer each one by identifier with one of three
verdicts: fixed, disputed with reasoning, or deferred with a reason. A report
that answers some findings and ignores others forces a second review.

## Frontend documentation

`static/index.html` starts with a block comment in the same shape as a Python
module header. Document the contract for data, not the markup: name the
endpoint each part of the page reads, the shape it expects, and what it
renders when that data is absent. The absent case is the one that breaks in
production.

Labels state what the software does. A value read by a model is presented as
a model read, with its flags, never as the official figure.

# !Important

Never commit any code.

# !Important

## Hyphenation

Do not compress a phrase into a hyphenated modifier placed in front of a noun.
If a hyphenated modifier could be rewritten with a preposition, meaning of,
for, from, at, in, or to, rewrite it that way. Prefer "code of the polling
station" to "polling-station-specific code". Prefer "reads of the vote
column" to "vote-column reads". Never chain three elements. Say "behaves
like a checksum" rather than "checksum-like". The test is whether a person
would say the phrase aloud in conversation, so keep only compounds already
established in the field, such as open source, by-election, or third party,
and drop the hyphen when the compound follows the noun rather than preceding
it. This rule applies to your own instructions, notes, and commentary as much
as to prose written for submission. Grammatical correctness does not justify
the construction. Writing that no person would produce is a defect even when
no rule has been broken.

## CODE FORMATING

Strictly follow https://peps.python.org/pep-0008/ 