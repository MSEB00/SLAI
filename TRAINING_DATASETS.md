# SLAI Training Dataset Shortlist

This list is focused on a personal assistant that needs:
- chat quality
- factual reliability
- preference alignment
- tool calling behavior

Always verify each dataset license and terms before training.

## 1) Instruction and Dialogue Data

- OpenAssistant OASST1
  - Link: https://huggingface.co/datasets/OpenAssistant/oasst1
  - Why: Human-written multi-turn assistant data.
  - Notes: Apache-2.0, 88.8k rows.

- Databricks Dolly 15k
  - Link: https://huggingface.co/datasets/HuggingFaceH4/databricks_dolly_15k
  - Why: Clean instruction-response pairs across task categories.
  - Notes: CC BY-SA 3.0, 15,015 rows.

- LMSYS Chat 1M
  - Link: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
  - Why: Real-world conversations at large scale.
  - Notes: Special dataset agreement required; includes moderation metadata and safety caveats.

## 2) Preference and Reward Data

- HelpSteer2
  - Link: https://huggingface.co/datasets/nvidia/HelpSteer2
  - Why: Multi-attribute ratings (helpfulness/correctness/coherence/complexity/verbosity).
  - Notes: CC BY 4.0, 21,362 rows.

- UltraFeedback
  - Link: https://github.com/OpenBMB/UltraFeedback
  - Why: Large preference dataset with rich feedback for reward modeling.
  - Notes: MIT, repo describes ~64k prompts and ~256k responses.

## 3) Tool Use and Function Calling

- ToolBench
  - Link: https://github.com/OpenBMB/ToolBench
  - Why: Tool-use planning/execution traces with API-centric tasks.
  - Notes: Apache-2.0, repo reports 126,486 instances and 16,464 APIs.

- Glaive Function Calling v2
  - Link: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
  - Why: Function-calling style supervised data.
  - Notes: Apache-2.0, 112,960 rows.

## 4) Hallucination and Factuality Evaluation

- TruthfulQA
  - Link: https://github.com/sylinrl/TruthfulQA
  - Why: Evaluate tendency to produce false but plausible answers.
  - Notes: Apache-2.0 benchmark repo with QA data and evaluation code.

- FEVER
  - Link: https://github.com/awslabs/fever
  - Why: Claim verification with evidence.
  - Notes: Apache-2.0, FEVER paper reports 185,441 claims.

- Natural Questions
  - Link: https://github.com/google-research-datasets/natural-questions
  - Why: Real search-query style QA grounded in Wikipedia evidence.
  - Notes: Apache-2.0.

## Suggested First Mix for SLAI

1. Start SFT with OASST1 + Dolly.
2. Add tool-calling stage with ToolBench + Glaive FC.
3. Add alignment stage using HelpSteer2 + UltraFeedback.
4. Evaluate every run on TruthfulQA + FEVER slices + targeted NQ set.

## Practical Advice for Your Hardware

Your laptop (RTX 3050 4 GB VRAM) is better for:
- data prep
- evaluation
- small adapter tuning in short runs

For reliable fine-tuning of 4B+ models, plan cloud GPU training.
