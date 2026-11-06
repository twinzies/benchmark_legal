# clc2-evaluation

A script to predict legislation from facts and claims using closed-source models, then evaluate the results.

**Authors:** Helena Xie, Tara K. Jain

## Quick start

1. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
```

2. Install dependencies (if needed):

```bash
pip install openai google-genai anthropic pydantic scikit-learn tqdm
```

3. Run one model:

```bash
# test run (uses sample cases)
python closed_source_eval.py --model GPT-5.5 --mode test-run

# full run (uses full test set)
python closed_source_eval.py --model GPT-5.5 --mode full-run
```

Note that in order to rum SaulLLM-{7 or 141}b-Instruct, you will need to self-host and serve an endpoint. This is currently configured through https://console.runpod.io/.

The .env file shared in this private repo points to the correct endpoint already with the model weights downloaded onto disk. The model costs ~$12/hour to host (and ~$4/day idle for the hardware rental).

Available `--model` values:
- `GPT-5.5`
- `o4`
- `Claude-Opus`
- `Gemini-flash`
- `Gemini-pro`
- `SaulLM-7B` (ignore for paper, only to validate the self-hosted vLLM setup)
- `SaulLM-141B`
- `Mixtral-141B` (base model of SaulLM-141B; served via OpenRouter)
- `GLM-5.2`
- `Qwen3.6-35B` (current best open Qwen)

4. Run all configured models via bash script:

```bash
bash run_all_evals.sh
```

The bash script runs `full-run` for all available models: `GPT-5.5`, `o4`, `Claude-Opus`, `Gemini-flash`, `Gemini-pro`, `SaulLM-7B`, `SaulLM-141B`, `Mixtral-141B`, `GLM-5.2`, and `Qwen3.6-35B`.
