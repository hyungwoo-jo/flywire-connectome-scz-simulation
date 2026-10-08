# Legacy (01-04): withdrawn interpretations

These are the first exploratory scripts (E/I perturbation, Powers-style conditioned hallucination, Bayesian prior learning, attractor landscape). They are kept as the original record only.

Their "hallucination / false memory" interpretations are **withdrawn**. Step 05 (`pipelines/05_connectome_null_control.py`, `docs/CONNECTOME_NULL_CONTROL.md`) showed that the large effects came from per-condition renormalization and were not specific to the real wiring, and that these scripts used mutable `root_id` annotations that do not match the v888 edge list.

Do not build on these scripts. The current work starts at step 06 and is summarized in `docs/RESEARCH_STATUS.md`.
