#!/usr/bin/env python3
"""Regenerate the 13 course notebooks from source-of-truth cell definitions.

Run once with `uv run python scripts/build_notebooks.py`. Keeping the notebooks
in Python (not hand-edited JSON) makes them reviewable in diffs and guarantees
valid nbformat + stable structure. Every executed code cell imports only
`omni_tutorial` + numpy + matplotlib (Agg-safe), so all notebooks run on a Mac
with no GPU. Official vLLM-Omni commands appear only in RAW cells (never executed).

Each notebook keeps the literal strings "Learning goals" and "Mac track" that
tests/test_course.py asserts on.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

MAC_TRACK = (
    "> **Mac track.** Every executed cell below is a pure-Python simulation that runs on "
    "any Mac with no GPU. Cells labelled **Linux GPU lab** show the *real* vLLM-Omni "
    "commands — they are shown as text and are **not** run here, because the official "
    "`vllm serve --omni` runtime is CUDA-only."
)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def raw(src):
    # RAW cells are never executed by nbclient — perfect for real GPU commands.
    return nbf.v4.new_raw_cell(src)


def header(title, goals):
    goal_lines = "\n".join(f"- {g}" for g in goals)
    return md(f"# {title}\n\n## Learning goals\n{goal_lines}\n\n{MAC_TRACK}")


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    # normalize: strip execution counts/outputs, ensure ids exist
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.write(nb, OUT / name)
    print(f"wrote {name} ({len(cells)} cells)")


# Plotting preamble. We do NOT force the Agg backend: under Jupyter the inline
# backend renders PNGs the learner sees, and under nbclient (make test) the
# default backend still executes cleanly headlessly. `%matplotlib inline` is a
# no-op outside IPython, so it is safe but we rely on Jupyter's default instead.
PLOT_SETUP = (
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
)


# ----------------------------------------------------------------------------- 00
def nb00():
    return [
        header("Start here: models, inference, and serving", [
            "Separate the model, runtime, server, and application layers",
            "Place vLLM-Omni precisely in that stack",
            "Learn the vocabulary the rest of the course uses",
        ]),
        md("## The four layers\n\n```text\n"
           "Application  -> voice assistant, image generator, robot agent\n"
           "Server       -> HTTP, streaming, admission control, metrics\n"
           "Runtime      -> scheduling, batching, memory, parallel execution\n"
           "Model        -> learned weights and architecture\n```\n\n"
           "**vLLM-Omni is the runtime + serving layer.** It does not train a model or add "
           "a modality the model never learned. It makes an existing multimodal model *serve* "
           "fast and cheap."),
        code(
            "layers = {\n"
            "    'model':       'Qwen3-Omni / Qwen-Image / Qwen3-TTS',\n"
            "    'runtime':     'vLLM-Omni (stages, connectors, orchestrator)',\n"
            "    'server':      'OpenAI-compatible HTTP API',\n"
            "    'application': 'your product',\n"
            "}\n"
            "for layer, value in layers.items():\n"
            "    print(f'{layer:12} -> {value}')"
        ),
        md("## Classify each thing\n\n"
           "Sort these into the four layers: `Qwen3-Omni`, `OmniConnector`, `/v1/audio/speech`, "
           "and a phone voice-assistant UI. Try it before running the next cell."),
        code(
            "answers = {\n"
            "    'Qwen3-Omni':        'model',\n"
            "    'OmniConnector':     'runtime',   # moves tensors between stages\n"
            "    '/v1/audio/speech':  'server',     # an HTTP endpoint\n"
            "    'phone voice UI':    'application',\n"
            "}\n"
            "for item, layer in answers.items():\n"
            "    print(f'{item:20} -> {layer}')"
        ),
        md("## Why a Mac course at all?\n\n"
           "The complete vLLM-Omni stack — heterogeneous AR/DiT stages, disaggregated serving — "
           "targets Linux + NVIDIA accelerators. You cannot run the production runtime here. But "
           "the *ideas* (paging, batching, stage graphs, queueing, guidance) are just accounting "
           "and linear algebra, and those run anywhere. This course teaches the ideas executably "
           "on your Mac, then points at the real source and shows the real commands."),
        md("## Source lab\n\n"
           "Open the real repo [`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni). "
           "Skim the top-level `vllm_omni/` package and the `README.md` model tables. Notice the split "
           "between `engine/`, `deploy/`, and `entrypoints/` — you will visit each later."),
    ]


# ----------------------------------------------------------------------------- 01
def nb01():
    return [
        header("From LLM to Omni: any-to-any shapes", [
            "Distinguish multimodal *input* from any-to-any *output*",
            "Recognize the three model families vLLM-Omni serves",
            "See why heterogeneous outputs force a pipeline, not a single decoder",
        ]),
        md("## Input vs output modality\n\n"
           "A plain LLM is text->text. A *multimodal-input* model is (text|image|audio)->text. "
           "An *any-to-any* model can also emit non-text: text->image, audio->audio, image->image.\n\n"
           "```text\n"
           "text-in            -> [LLM]        -> text-out\n"
           "text+image+audio   -> [Omni model] -> text | image | audio\n```\n\n"
           "Emitting audio or pixels is not one more decoding step — it needs a *different kind* of "
           "stage (an audio codec, a diffusion decoder). That is the whole reason vLLM-Omni exists."),
        code(
            "families = {\n"
            "    'Omni-modality': ['Qwen3-Omni', 'BAGEL', 'Cosmos', 'HunyuanImage'],\n"
            "    'TTS':           ['Qwen3-TTS', 'CosyVoice3', 'VoxCPM2'],\n"
            "    'Diffusion':     ['Qwen-Image', 'Wan2.2', 'FLUX'],\n"
            "}\n"
            "for family, models in families.items():\n"
            "    print(f'{family:14}: ' + ', '.join(models))"
        ),
        md("## The output shape drives the architecture\n\n"
           "Below we tag a few example models with their input and output modalities. Notice that "
           "the moment an output is `image` or `audio`, the runtime needs a non-AR stage to produce it."),
        code(
            "examples = [\n"
            "    ('Qwen3-Omni', 'text+image+audio', 'text+audio'),\n"
            "    ('FLUX',       'text',             'image'),\n"
            "    ('Qwen3-TTS',  'text',             'audio'),\n"
            "]\n"
            "for model, inp, out in examples:\n"
            "    needs_nonar = any(m in out for m in ('image', 'audio'))\n"
            "    print(f'{model:12} {inp:18} -> {out:12} | non-AR stage needed: {needs_nonar}')"
        ),
        md("## Exercise\n\n"
           "A product needs speech-in, speech-out translation. Which family, and how many *kinds* of "
           "stage (AR text, AR audio-codec, codec-to-wave) does the output path imply? Write your guess, "
           "then run the solution."),
        code(
            "# Solution\n"
            "print('Family: Omni-modality (understands audio, emits audio).')\n"
            "print('Output path stages: Thinker (AR text) -> Talker (AR audio codes) -> Code2Wav.')\n"
            "print('That is the exact Qwen3-Omni speech pipeline you meet in notebook 03.')"
        ),
        md("## Source lab\n\nRead the supported-model tables in the repo `README.md`. For one model in each "
           "family, find its config under `vllm_omni/deploy/` and note how many stages it declares."),
    ]


# ----------------------------------------------------------------------------- 02
def nb02():
    return [
        header("vLLM foundations: KV cache and continuous batching", [
            "Compute KV-cache memory and see block paging fight fragmentation",
            "Contrast static vs continuous batching on the same workload",
            "Understand what AR stages inherit for free from vLLM",
        ]),
        md("## Why AR generation is memory-bound\n\n"
           "Every generated token must attend to all previous tokens, so their keys and values are "
           "cached. That KV cache grows with sequence length and dominates accelerator memory — which "
           "is what caps how many requests run at once. vLLM's **PagedAttention** carves the cache into "
           "fixed **blocks** and allocates them lazily, so sequences share memory instead of "
           "pre-reserving their maximum length."),
        code(
            "from omni_tutorial import kv_bytes_per_token, PagedKVCache\n"
            "# A small model's per-token footprint across all layers:\n"
            "per_tok = kv_bytes_per_token(num_layers=28, num_kv_heads=4, head_dim=128, dtype='fp16')\n"
            "print(f'{per_tok:,} bytes/token  ({per_tok/1024:.1f} KiB)')\n"
            "seq = 2048\n"
            "print(f'one 2048-token sequence: {per_tok*seq/1e6:.1f} MB of KV cache')"
        ),
        md("## Paging and fragmentation\n\n"
           "We grow a few sequences token-by-token in a fixed block pool and watch utilization. The gap "
           "between allocated and used slots is *internal fragmentation* — wasted space in each "
           "sequence's partially-filled last block."),
        code(
            "cache = PagedKVCache(total_blocks=64, block_size=16)\n"
            "import itertools\n"
            "seqs = ['a', 'b', 'c']\n"
            "used, wasted = [], []\n"
            "for step in range(40):\n"
            "    for s in seqs:\n"
            "        cache.append(s, 1)\n"
            "    used.append(cache.used_slots)\n"
            "    wasted.append(cache.wasted_slots)\n"
            "print(f'after 40 steps: {cache.used_slots} slots used, {cache.wasted_slots} wasted, '\n"
            "      f'utilization {cache.utilization:.0%}')"
        ),
        code(
            PLOT_SETUP +
            "fig, ax = plt.subplots(figsize=(6, 3.2))\n"
            "steps = range(1, len(used) + 1)\n"
            "ax.stackplot(steps, used, wasted, labels=['used slots', 'wasted (fragmentation)'],\n"
            "             colors=['#4c78a8', '#f2cf5b'])\n"
            "ax.set_xlabel('decode step'); ax.set_ylabel('KV slots')\n"
            "ax.set_title('Paged KV cache: used vs fragmented slots'); ax.legend(loc='upper left')\n"
            "fig.tight_layout(); plt.show()"
        ),
        md("## Static vs continuous batching\n\n"
           "One long request shares a batch with several short ones. Static batching locks the batch until "
           "the straggler finishes; continuous batching refills freed slots immediately."),
        code(
            "from omni_tutorial import (Request, simulate_static_batching,\n"
            "                           simulate_continuous_batching, throughput)\n"
            "reqs = [Request('long', 8, 12)] + [Request(f's{i}', 8, 2) for i in range(5)]\n"
            "static = simulate_static_batching(reqs, max_batch=3)\n"
            "cont = simulate_continuous_batching(reqs, max_batch=3)\n"
            "print(f'static:     {len(static):2d} iters, mean batch {throughput(static):.2f}')\n"
            "print(f'continuous: {len(cont):2d} iters, mean batch {throughput(cont):.2f}')"
        ),
        code(
            PLOT_SETUP +
            "fig, ax = plt.subplots(figsize=(6, 3.2))\n"
            "ax.step(range(len(static)), static, where='post', label='static')\n"
            "ax.step(range(len(cont)), cont, where='post', label='continuous')\n"
            "ax.set_xlabel('iteration'); ax.set_ylabel('live sequences in batch')\n"
            "ax.set_title('Continuous batching keeps slots full'); ax.legend()\n"
            "fig.tight_layout(); plt.show()"
        ),
        md("## Exercise\n\n"
           "Double `block_size` to 32 in the paging cell. Do you expect more or less wasted space per "
           "sequence? Run it and check against the solution."),
        code(
            "# Solution: bigger blocks -> coarser allocation -> MORE potential waste per sequence\n"
            "for bs in (8, 16, 32):\n"
            "    c = PagedKVCache(total_blocks=256, block_size=bs)\n"
            "    c.append('x', 17)  # 17 tokens\n"
            "    print(f'block_size={bs:2d}: {c.wasted_slots} wasted slots for a 17-token sequence')"
        ),
        raw(
            "# Linux GPU lab (not run here) — the real knobs behind this notebook:\n"
            "vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --block-size 16 \\\n"
            "    --gpu-memory-utilization 0.90 \\\n"
            "    --max-num-seqs 256\n"
            "# larger --max-num-seqs = bigger continuous batch, bounded by KV memory."
        ),
        md("## Source lab\n\nFind the KV-cache block manager and the scheduler in the vLLM core that "
           "vLLM-Omni reuses for AR stages. Trace how a finished sequence's blocks return to the pool."),
    ]


# ----------------------------------------------------------------------------- 03
def nb03():
    return [
        header("Autoregressive stages: Thinker and Talker", [
            "Trace one request through Thinker -> Talker -> Code2Wav",
            "See hidden states become an inter-stage payload",
            "Understand why the runner re-applies a preprocess every decode step",
        ]),
        md("## The Qwen-Omni speech path\n\n"
           "```text\n"
           "multimodal input -> Thinker (AR text) -> text + hidden states\n"
           "                 -> Talker  (AR audio codes) -> Code2Wav -> waveform\n```\n\n"
           "Thinker and Talker are *both* autoregressive, but their vocabularies differ: Thinker emits "
           "text tokens, Talker emits audio-codec tokens conditioned on Thinker's hidden states."),
        code(
            "from omni_tutorial import build_voice_pipeline\n"
            "graph = build_voice_pipeline()\n"
            "result, trace = graph.run('What is in this scene?')\n"
            "for e in trace:\n"
            "    print(f\"{e['stage']:8} ({e['kind']:5}) -> {e['output']}\")\n"
            "print('connector transfers:', graph.connector.transfers)"
        ),
        md("## Hidden states are an internal dependency\n\n"
           "Thinker's hidden representation is not a client response — it is conditioning for the Talker. "
           "A model-specific **preprocess** function selects, projects, and reshapes it. Crucially, the "
           "real vLLM-Omni model runner re-applies this preprocess *every decode iteration*, because the "
           "Talker fuses fresh Thinker output with its own running state at each step."),
        code(
            "import numpy as np\n"
            "rng = np.random.default_rng(0)\n"
            "def talker_preprocess(thinker_hidden, step):\n"
            "    # project hidden state and add a per-step positional nudge (toy stand-in)\n"
            "    proj = thinker_hidden @ rng.standard_normal((thinker_hidden.shape[-1], 4))\n"
            "    return proj + step * 0.01\n"
            "hidden = rng.standard_normal((1, 8))\n"
            "for step in range(3):\n"
            "    cond = talker_preprocess(hidden, step)\n"
            "    print(f'decode step {step}: conditioning shape {cond.shape}, mean {cond.mean():+.3f}')"
        ),
        md("## Text tokens vs audio-codec tokens\n\n"
           "The two AR stages produce sequences of different lengths and meanings. On video-input tasks "
           "the paper measures ~150 text tokens but ~545 audio tokens per request — the Talker simply "
           "runs far more decode steps. Remember that number; it is why the Talker is the bottleneck "
           "you scale in notebook 08."),
        code(
            "text_tokens, audio_tokens = 150, 545\n"
            "print(f'Talker runs {audio_tokens/text_tokens:.1f}x more decode steps than the Thinker text head')"
        ),
        md("## Exercise\n\n"
           "Using the trace above, which stage's output is the *client-facing* one, and which is purely "
           "inter-stage? Answer, then run the solution."),
        code(
            "# Solution\n"
            "print('Inter-stage: thinker hidden states, talker audio codes (internal payloads).')\n"
            "print('Client-facing: the vocoder waveform (and optionally the thinker text transcript).')"
        ),
        raw(
            "# Linux GPU lab (not run here) — launch the three AR/codec stages as separate processes:\n"
            "CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --port 8091 --stage-id 0 --omni-master-address 127.0.0.1 --omni-master-port 26000\n"
            "CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --stage-id 1 --headless --omni-master-address 127.0.0.1 --omni-master-port 26000\n"
            "CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --stage-id 2 --headless --omni-master-address 127.0.0.1 --omni-master-port 26000"
        ),
        md("## Source lab\n\nTrace `vllm_omni/model_executor/models/qwen3_omni/` and its deploy config "
           "`deploy/qwen3_omni_moe.yaml`. Find where the Talker's per-step preprocess consumes Thinker output."),
    ]


# ----------------------------------------------------------------------------- 04
def nb04():
    return [
        header("Diffusion stages: DiT, CFG, latents, and VAE", [
            "Run a denoising loop over latents and watch it converge",
            "See classifier-free guidance change the trajectory",
            "Map latent shapes to pixel shapes through a VAE",
        ]),
        md("## Diffusion is not autoregressive\n\n"
           "A DiT stage starts from pure noise in a compressed **latent** space and iteratively denoises "
           "toward a sample. There is no KV cache and no token-by-token loop — instead, a fixed number of "
           "denoising steps each run a full forward pass. A **VAE** then decodes the final latent to pixels.\n\n"
           "```text\nnoise latent -> [DiT x N steps] -> clean latent -> [VAE decode] -> image\n```"),
        code(
            "from omni_tutorial import denoise, distance_to_target, vae_decode_shape\n"
            "import numpy as np\n"
            "rng = np.random.default_rng(1)\n"
            "latent = rng.standard_normal((8, 8, 4))   # (h, w, channels) in latent space\n"
            "target = rng.standard_normal((8, 8, 4))   # the sample the toy 'model' pulls toward\n"
            "traj = denoise(latent, target, steps=25, cfg_scale=1.0)\n"
            "print(f'{len(traj)-1} denoising steps; final distance to target '\n"
            "      f'{distance_to_target(traj, target)[-1]:.3f}')"
        ),
        md("## Classifier-free guidance\n\n"
           "CFG combines a conditional and unconditional prediction: "
           "`guided = uncond + scale * (cond - uncond)`. A scale above 1 extrapolates past the "
           "conditional prediction, pushing harder toward the prompt. We sweep the scale and plot how "
           "fast each run approaches the target."),
        code(
            PLOT_SETUP +
            "from omni_tutorial import denoise, distance_to_target\n"
            "fig, ax = plt.subplots(figsize=(6, 3.4))\n"
            "for scale in (1.0, 2.0, 4.0):\n"
            "    traj = denoise(latent, target, steps=25, cfg_scale=scale)\n"
            "    ax.plot(distance_to_target(traj, target), label=f'cfg={scale}')\n"
            "ax.set_xlabel('denoising step'); ax.set_ylabel('L2 distance to target')\n"
            "ax.set_title('Guidance strength vs convergence'); ax.legend()\n"
            "fig.tight_layout(); plt.show()"
        ),
        md("## Latent -> pixel through the VAE\n\n"
           "The DiT works in a spatially compressed latent (typically 8x smaller per axis). The VAE "
           "decoder expands it back to RGB pixels."),
        code(
            "for latent_shape in [(8, 8, 4), (64, 64, 4), (96, 160, 4)]:\n"
            "    print(f'latent {latent_shape} -> image {vae_decode_shape(latent_shape)}')"
        ),
        md("## Exercise\n\n"
           "At very high CFG the curve can overshoot and rise again. Predict at which scale, then sweep "
           "`cfg_scale` in the plot cell to find where convergence stops improving."),
        code(
            "# Solution: extrapolation past the target eventually overshoots.\n"
            "for scale in (1, 2, 4, 8):\n"
            "    final = distance_to_target(denoise(latent, target, steps=25, cfg_scale=scale), target)[-1]\n"
            "    print(f'cfg={scale}: final distance {final:.3f}')"
        ),
        raw(
            "# Linux GPU lab (not run here) — a single-stage DiT model (BAGEL is stage-0-only):\n"
            "vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091\n"
            "# the DiT stage contains the LLM, ViT, VAE and tokenizer internally."
        ),
        md("## Source lab\n\nRead a diffusion model's deploy config in `vllm_omni/deploy/` and find its "
           "scheduler / step count and where the VAE decode happens. Compare to the AR configs from nb03."),
    ]


# ----------------------------------------------------------------------------- 05
def nb05():
    return [
        header("Stage graphs: the core vLLM-Omni abstraction", [
            "Read a pipeline as nodes (stages) and edges (transfer functions)",
            "Build and run a custom stage graph",
            "See where transfer functions transform inter-stage data",
        ]),
        md("## Nodes and edges\n\n"
           "vLLM-Omni models any pipeline as a graph: **nodes** are stages (an AR LLM, a DiT), **edges** "
           "carry data between them through a connector, optionally applying a **transfer function** that "
           "reshapes one stage's output into the next stage's input. The voice pipeline is the linear case "
           "`Thinker -> Talker -> Code2Wav`, but the abstraction is general."),
        code(
            "from omni_tutorial import Stage, StageGraph, Connector\n"
            "# Build a custom 2-stage graph with an explicit transfer function on the edge.\n"
            "encode = Stage('encode', 'AR', lambda text: {'tokens': text.split()})\n"
            "def to_summary(payload):\n"
            "    return {'summary': ' '.join(payload['tokens'][:3]) + ' ...'}\n"
            "summarize = Stage('summarize', 'AR', lambda p: {'result': to_summary(p)})\n"
            "graph = StageGraph([encode, summarize])\n"
            "out, trace = graph.run('the quick brown fox jumps')\n"
            "for e in trace:\n"
            "    print(f\"{e['stage']:10} -> {e['output']}\")"
        ),
        md("## Every stage carries a cost\n\n"
           "Each `Stage` has a `service_time`. The graph can project itself onto stage specs for the "
           "performance simulator you use in notebook 08 — this is the bridge from *structure* to "
           "*performance*."),
        code(
            "from omni_tutorial import build_voice_pipeline\n"
            "specs = build_voice_pipeline().stage_specs()\n"
            "for s in specs:\n"
            "    print(f'{s.name:8} service_time={s.service_time}  replicas={s.replicas}')"
        ),
        md("## Exercise\n\n"
           "Add a third stage `translate` after `encode` that reverses the token list, and run the graph. "
           "How many connector transfers do you expect for 3 stages?"),
        code(
            "# Solution: N stages -> N-1 edges -> N-1 transfers.\n"
            "from omni_tutorial import Stage, StageGraph\n"
            "translate = Stage('translate', 'AR', lambda p: {'tokens': list(reversed(p['tokens']))})\n"
            "g = StageGraph([encode, translate, summarize])\n"
            "out, trace = g.run('the quick brown fox jumps')\n"
            "print('stages:', [e['stage'] for e in trace])\n"
            "print('transfers:', g.connector.transfers)  # expect 2"
        ),
        md("## Source lab\n\nFind the stage-graph / deployment abstraction in `vllm_omni/` (look under "
           "`engine/` and `deploy/`). Identify what a node and an edge map to in code, and where a transfer "
           "function is registered."),
    ]


# ----------------------------------------------------------------------------- 06
def nb06():
    return [
        header("Request lifecycle and orchestration", [
            "Follow a request through the orchestrator state machine",
            "See one public request fan out into per-stage requests",
            "Distinguish inter-stage outputs from final client outputs",
        ]),
        md("## Lifecycle\n\n"
           "```text\n"
           "HTTP/offline request\n"
           " -> normalize input\n"
           " -> AsyncOmniEngine\n"
           " -> Orchestrator creates request state\n"
           " -> stage 0 engine -> output processor -> connector\n"
           " -> stage 1 engine -> ... -> final OmniRequestOutput / stream\n```\n\n"
           "The orchestrator is a control-plane component: it schedules stages and routes data, but does "
           "not execute neural layers itself. Each stage has its own engine, scheduler, and KV manager."),
        code(
            "STATES = ['RECEIVED', 'STAGE_0_RUNNING', 'TRANSFER_0_1', 'STAGE_1_RUNNING',\n"
            "          'TRANSFER_1_2', 'STAGE_2_RUNNING', 'COMPLETED']\n"
            "for i, s in enumerate(STATES):\n"
            "    print(f'{i}  {s}')"
        ),
        md("## Request state: one public id, many stage ids\n\n"
           "A single client request can spawn several stage requests (and companion branches). The "
           "orchestrator tracks their ids, dependencies, streaming status, cancellation, and which outputs "
           "are *final* vs *internal*."),
        code(
            "def run_lifecycle(public_id, n_stages, cancel_at=None):\n"
            "    req = {'public_id': public_id, 'stage_ids': {}, 'status': 'RECEIVED',\n"
            "           'final_outputs': []}\n"
            "    for stage in range(n_stages):\n"
            "        if cancel_at == stage:\n"
            "            req['status'] = f'CANCELLED_AT_STAGE_{stage}'\n"
            "            return req\n"
            "        req['stage_ids'][stage] = f'{public_id}-stage-{stage}'\n"
            "        req['status'] = f'STAGE_{stage}_RUNNING'\n"
            "    req['status'] = 'COMPLETED'\n"
            "    req['final_outputs'] = ['audio.wav']  # only the last stage is client-facing\n"
            "    return req\n"
            "print(run_lifecycle('req-7', 3))\n"
            "print(run_lifecycle('req-8', 3, cancel_at=1))"
        ),
        md("## Why streaming decouples the stages\n\n"
           "The output processor asynchronously streams partial output to the next stage, so the Talker "
           "can start before the Thinker fully finishes. This is what reduces time-to-first-token."),
        code(
            "# toy: thinker emits tokens; talker consumes them as they arrive\n"
            "thinker_stream = iter(['The', 'cat', 'sat', 'down'])\n"
            "for i, tok in enumerate(thinker_stream):\n"
            "    print(f't={i}: thinker emitted {tok!r} -> talker can begin decoding on it')"
        ),
        md("## Exercise\n\n"
           "Modify `run_lifecycle` to record a timestamp per state transition (use a simple counter). "
           "Which transition would you watch to measure time-to-first-token?"),
        code(
            "# Solution: TTFT is the time from RECEIVED to the FIRST client-visible token,\n"
            "# which (with streaming) happens as soon as the LAST stage emits its first output,\n"
            "# not when the whole pipeline COMPLETES.\n"
            "print('Watch: RECEIVED -> first emit from the final (client-facing) stage.')"
        ),
        raw(
            "# Linux GPU lab (not run here) — the unified launch runs orchestrator + all stages:\n"
            "vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091\n"
            "# the default deploy config vllm_omni/deploy/qwen3_omni_moe.yaml wires the stages."
        ),
        md("## Source lab\n\nRead `entrypoints/omni.py` -> `engine/async_omni_engine.py` -> "
           "`engine/orchestrator.py`. Find request-state creation, stage submission, output consumption, "
           "and cancellation."),
    ]


# ----------------------------------------------------------------------------- 07
def nb07():
    return [
        header("Connectors and disaggregation", [
            "Understand what the OmniConnector transports and why",
            "Compare shared-memory vs Mooncake transport with real numbers",
            "See why connector overhead is negligible next to inference",
        ]),
        md("## The OmniConnector decouples transport from model logic\n\n"
           "Stages must hand each other embeddings, hidden states, audio codes, and image tensors. The "
           "**OmniConnector** generalizes vLLM's KV-cache transfer into one put/get interface over any "
           "transport:\n\n"
           "- **single-node:** inline control queues for small payloads, shared memory for large ones\n"
           "- **multi-node:** a Mooncake-based connector over TCP or RDMA, passing only lightweight "
           "metadata through the control plane\n\n"
           "This is what makes **full disaggregation** possible: each stage runs on its own engine with "
           "its own accelerators, and the connector moves data between them."),
        code(
            "# Measured connector latencies from the paper (Qwen2.5-Omni), in milliseconds:\n"
            "overhead = {\n"
            "    'Thinker2Talker': {'shared_memory': 5.49, 'mooncake': 8.28},\n"
            "    'Talker2Vocoder': {'shared_memory': 0.53, 'mooncake': 3.34},\n"
            "}\n"
            "for edge, modes in overhead.items():\n"
            "    print(f\"{edge:16} shared={modes['shared_memory']:.2f} ms  mooncake={modes['mooncake']:.2f} ms\")"
        ),
        code(
            PLOT_SETUP +
            "edges = ['Thinker2Talker', 'Talker2Vocoder']\n"
            "shared = [5.49, 0.53]; mooncake = [8.28, 3.34]\n"
            "x = np.arange(len(edges)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(6, 3.2))\n"
            "ax.bar(x - w/2, shared, w, label='shared memory (1 node)')\n"
            "ax.bar(x + w/2, mooncake, w, label='Mooncake (multi-node)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(edges)\n"
            "ax.set_ylabel('transfer latency (ms)')\n"
            "ax.set_title('Connector overhead by transport'); ax.legend()\n"
            "fig.tight_layout(); plt.show()"
        ),
        md("## Overhead in context\n\n"
           "End-to-end omni inference takes *tens of seconds*. A few milliseconds of transfer per edge is "
           "negligible — which is exactly why disaggregating stages across nodes is a good trade."),
        code(
            "inference_s = 20.0\n"
            "worst_transfer_ms = 8.28 + 3.34\n"
            "print(f'connector overhead is {worst_transfer_ms/1000/inference_s:.4%} of a 20s request')"
        ),
        md("## Exercise\n\n"
           "If you scale the Talker onto a second node, which connector edge changes transport, and by how "
           "many ms? Compute the added latency."),
        code(
            "# Solution: Thinker->Talker crosses the node boundary and switches to Mooncake.\n"
            "added = 8.28 - 5.49\n"
            "print(f'Thinker2Talker goes shared->mooncake: +{added:.2f} ms (still trivial vs seconds)')"
        ),
        raw(
            "# Linux GPU lab (not run here) — swap in a network connector for multi-node:\n"
            "# in the deploy overlay YAML, set the stage edges to a MooncakeStoreConnector\n"
            "# and give each stage output_connectors / input_connectors (to_stage_1, from_stage_0, ...)."
        ),
        md("## Source lab\n\nFind the connector interface in `vllm_omni/` and the shared-memory and Mooncake "
           "implementations. Locate the put/get calls and what metadata rides the control plane."),
    ]


# ----------------------------------------------------------------------------- 08
def nb08():
    return [
        header("Scheduling, batching, and scaling", [
            "Find the bottleneck stage in a heterogeneous pipeline",
            "Scale a stage's replicas and measure the makespan drop",
            "See diminishing returns once the bottleneck moves",
        ]),
        md("## The slowest stage sets the throughput\n\n"
           "In a pipeline, the stage with the lowest sustained throughput (replicas / service_time) is the "
           "bottleneck. For Qwen3-Omni that is the **Talker**: it decodes ~3.6x more tokens than the text "
           "head. We feed the voice pipeline's service times into a tandem-queue simulator and locate it."),
        code(
            "from omni_tutorial import build_voice_pipeline, PipelineSimulator\n"
            "specs = build_voice_pipeline().stage_specs()\n"
            "arrivals = [i * 0.5 for i in range(40)]\n"
            "result = PipelineSimulator(specs).run(arrivals)\n"
            "print('bottleneck:', result['bottleneck'])\n"
            "for s in result['stage_stats']:\n"
            "    flag = '  <-- bottleneck' if s['bottleneck'] else ''\n"
            "    print(f\"{s['stage']:8} mean_wait={s['mean_wait']:5.1f} capacity={s['capacity']:.2f}{flag}\")"
        ),
        md("## Scale the bottleneck\n\n"
           "We sweep the Talker's replica count and watch the makespan (time to clear all 40 requests) "
           "and the Talker's own max wait fall — then flatten once a different stage becomes limiting."),
        code(
            "from omni_tutorial import sweep_replicas\n"
            "rows = sweep_replicas(specs, 'talker', [1, 2, 3, 4, 5, 6], arrivals)\n"
            "for r in rows:\n"
            "    print(f\"talker x{r['replicas']}: makespan={r['makespan']:6.1f} \"\n"
            "          f\"talker_max_wait={r['stage_max_wait']:6.1f} bottleneck={r['bottleneck']}\")"
        ),
        code(
            PLOT_SETUP +
            "from omni_tutorial import sweep_replicas\n"
            "rows = sweep_replicas(specs, 'talker', list(range(1, 9)), arrivals)\n"
            "reps = [r['replicas'] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(6, 3.2))\n"
            "ax.plot(reps, [r['makespan'] for r in rows], 'o-', label='makespan')\n"
            "ax.plot(reps, [r['stage_max_wait'] for r in rows], 's--', label='talker max wait')\n"
            "ax.set_xlabel('talker replicas'); ax.set_ylabel('time (arb. units)')\n"
            "ax.set_title('Scaling the bottleneck: diminishing returns'); ax.legend()\n"
            "fig.tight_layout(); plt.show()"
        ),
        md("## Exercise\n\n"
           "After enough Talker replicas the bottleneck moves. Which stage does it move to, and why doesn't "
           "adding *more* Talkers help past that point?"),
        code(
            "# Solution: once talker capacity exceeds the others, the Thinker (service_time=1.0,\n"
            "# 1 replica) becomes limiting. Extra talkers then sit idle waiting for thinker output.\n"
            "rows = sweep_replicas(specs, 'talker', [1, 4, 8], arrivals)\n"
            "for r in rows:\n"
            "    print(f\"talker x{r['replicas']}: bottleneck is now {r['bottleneck']}\")"
        ),
        raw(
            "# Linux GPU lab (not run here) — scale specific stages with per-stage overrides:\n"
            "vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \\\n"
            "    --stage-overrides '{\"1\": {\"num_replicas\": 2, \"devices\": \"1,2\"},\n"
            "                        \"2\": {\"num_replicas\": 2, \"devices\": \"1,2\"}}'"
        ),
        md("## Source lab\n\nFind per-stage scheduling and the `--stage-overrides` handling in "
           "`vllm_omni/`. Confirm each stage engine owns its own scheduler and replica set."),
    ]


# ----------------------------------------------------------------------------- 09
def nb09():
    return [
        header("APIs and streaming", [
            "Contrast the offline engine API with the OpenAI-style HTTP API",
            "Recognize the real media endpoints vLLM-Omni exposes",
            "Simulate a streamed response and reason about TTFT",
        ]),
        md("## Two ways in\n\n"
           "- **Offline**: construct an engine in-process and call it with a batch of prompts (great for "
           "benchmarking and pipelines).\n"
           "- **Online**: an OpenAI-compatible HTTP server with familiar endpoints, so existing clients "
           "work with minimal changes.\n\n"
           "vLLM-Omni adds media endpoints on top of chat/completions, e.g. `/v1/audio/speech` and "
           "image/video generation routes."),
        code(
            "endpoints = {\n"
            "    '/v1/chat/completions': 'text + multimodal input -> text (+ optional audio)',\n"
            "    '/v1/audio/speech':     'text -> audio (TTS path: Talker + Code2Wav)',\n"
            "    '/v1/images/generations': 'text -> image (DiT stage)',\n"
            "}\n"
            "for route, desc in endpoints.items():\n"
            "    print(f'{route:26} {desc}')"
        ),
        md("## Simulating a streamed chat response\n\n"
           "Streaming yields tokens as they are produced. The first token's arrival is the "
           "time-to-first-token (TTFT); the gap between tokens is the inter-token latency."),
        code(
            "def stream(tokens, ttft, itl):\n"
            "    t = ttft\n"
            "    for tok in tokens:\n"
            "        yield round(t, 2), tok\n"
            "        t += itl\n"
            "for arrival, tok in stream(['Hello', 'from', 'vLLM', '-', 'Omni'], ttft=0.30, itl=0.05):\n"
            "    print(f't={arrival:4.2f}s  {tok}')"
        ),
        md("## Exercise\n\n"
           "A client shows text and speaks it. With streaming, should it wait for the whole response before "
           "starting TTS, or pipe tokens through as they arrive? Justify in terms of TTFT."),
        code(
            "# Solution: pipe them through. Streaming to the Talker/TTS as tokens arrive means audio can\n"
            "# start near the text TTFT instead of after the full text completes -- the same decoupling\n"
            "# the orchestrator uses between stages (notebook 06).\n"
            "print('Stream through: perceived latency ~ text TTFT, not full-response time.')"
        ),
        raw(
            "# Linux GPU lab (not run here) — call the running server, OpenAI-style:\n"
            "curl http://127.0.0.1:8091/v1/audio/speech \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"model\": \"Qwen/Qwen3-Omni-30B-A3B-Instruct\", \"input\": \"hello world\", \"voice\": \"default\"}'"
        ),
        md("## Source lab\n\nRead the OpenAI-compatible server entrypoints in `vllm_omni/entrypoints/`. "
           "Match each media endpoint to the stage path it drives."),
    ]


# ----------------------------------------------------------------------------- 10
def nb10():
    return [
        header("Observability and benchmarking", [
            "Define RTF, JCT, TTFT and compute them on simulated data",
            "Avoid the classic benchmarking pitfalls",
            "Read the paper's headline speedups as targets, not magic",
        ]),
        md("## The metrics that matter\n\n"
           "- **TTFT** — time to first token/output; dominates *perceived* latency.\n"
           "- **JCT** — job completion time; the full request end to end.\n"
           "- **RTF** — real-time factor for audio: generation_time / audio_duration. RTF < 1 means "
           "faster-than-real-time (good for live speech).\n\n"
           "We compute all three on a simulated request set."),
        code(
            "requests = [\n"
            "    {'ttft': 0.30, 'jct': 4.2, 'audio_s': 6.0},\n"
            "    {'ttft': 0.41, 'jct': 5.1, 'audio_s': 5.5},\n"
            "    {'ttft': 0.28, 'jct': 3.9, 'audio_s': 7.2},\n"
            "]\n"
            "def rtf(r): return r['jct'] / r['audio_s']\n"
            "mean = lambda xs: sum(xs) / len(xs)\n"
            "print(f\"mean TTFT {mean([r['ttft'] for r in requests]):.2f}s\")\n"
            "print(f\"mean JCT  {mean([r['jct'] for r in requests]):.2f}s\")\n"
            "print(f\"mean RTF  {mean([rtf(r) for r in requests]):.2f}  (<1 is real-time)\")"
        ),
        md("## Pitfalls that make you lie to yourself\n\n"
           "1. **Reporting the mean, hiding the tail.** p99 latency is what users feel. Plot the "
           "distribution.\n"
           "2. **Warm vs cold.** The first request pays compilation/allocation costs; drop warmup runs.\n"
           "3. **Batch size of one.** Serving throughput comes from batching; a single-request benchmark "
           "measures the wrong thing."),
        code(
            PLOT_SETUP +
            "rng = np.random.default_rng(3)\n"
            "# a realistic latency sample: mostly fast, with a heavy tail\n"
            "latencies = np.concatenate([rng.normal(0.4, 0.05, 480), rng.normal(1.2, 0.3, 20)])\n"
            "p50, p99 = np.percentile(latencies, [50, 99])\n"
            "fig, ax = plt.subplots(figsize=(6, 3.2))\n"
            "ax.hist(latencies, bins=40, color='#4c78a8')\n"
            "ax.axvline(p50, color='green', label=f'p50={p50:.2f}s')\n"
            "ax.axvline(p99, color='red', label=f'p99={p99:.2f}s')\n"
            "ax.set_xlabel('latency (s)'); ax.set_ylabel('requests')\n"
            "ax.set_title('Mean hides the tail'); ax.legend()\n"
            "fig.tight_layout(); plt.show()\n"
            "print(f'mean={latencies.mean():.2f}s but p99={p99:.2f}s')"
        ),
        md("## Targets from the paper\n\n"
           "Disaggregated vLLM-Omni reports large wins over a monolithic baseline. Treat these as what "
           "*good* looks like when the bottleneck stage is properly scaled."),
        code(
            "targets = {\n"
            "    'Qwen3-Omni': 'RTF -90.7%, JCT -91.4%, Thinker TPS 12.97x, Talker TPS 7.98x',\n"
            "    'BAGEL':      'T2I 2.40x, I2I 3.72x',\n"
            "    'MiMo-Audio': 'RTF 1.39 -> 0.12 with graph compilation (11.58x)',\n"
            "}\n"
            "for model, wins in targets.items():\n"
            "    print(f'{model:12}: {wins}')"
        ),
        md("## Exercise\n\n"
           "Given the latency sample above, would reporting only the mean overstate or understate the "
           "user experience? Compute the gap."),
        code(
            "# Solution\n"
            "print(f'p99 is {p99/latencies.mean():.1f}x the mean -- reporting the mean understates tail pain.')"
        ),
        md("## Source lab\n\nRun the repo's `benchmarks/` against a served model on Linux. Compare your RTF "
           "and JCT to the targets above and see which stage limits you."),
    ]


# ----------------------------------------------------------------------------- 11
def nb11():
    return [
        header("Adding a model: the integration surface", [
            "Enumerate what a new model must declare to vLLM-Omni",
            "Sketch a deploy config with stages and connector edges",
            "Understand the per-stage preprocess/transfer contract",
        ]),
        md("## What integration actually requires\n\n"
           "Adding a model is mostly *declaration*, not new serving code: you describe its stages, wire "
           "the edges, and supply the per-stage preprocess/transfer functions. The runtime (scheduling, "
           "batching, KV management, connectors) is reused."),
        code(
            "# A minimal deploy config as a Python dict (real ones are YAML under vllm_omni/deploy/).\n"
            "deploy = {\n"
            "    'model': 'my-org/my-omni-model',\n"
            "    'stages': [\n"
            "        {'id': 0, 'role': 'thinker', 'kind': 'AR',    'output_connectors': ['to_stage_1']},\n"
            "        {'id': 1, 'role': 'talker',  'kind': 'AR',    'input_connectors': ['from_stage_0'],\n"
            "                                                        'output_connectors': ['to_stage_2']},\n"
            "        {'id': 2, 'role': 'code2wav','kind': 'codec', 'input_connectors': ['from_stage_1']},\n"
            "    ],\n"
            "}\n"
            "import json; print(json.dumps(deploy, indent=2))"
        ),
        md("## The preprocess/transfer contract\n\n"
           "Each stage supplies a function that turns the previous stage's payload into its own input. The "
           "runner calls it every iteration for AR stages that fuse upstream output at each step. Below we "
           "validate a toy contract: the function must accept the upstream payload and return the keys the "
           "stage declares it needs."),
        code(
            "def make_transfer(required_keys):\n"
            "    def transfer(payload):\n"
            "        missing = [k for k in required_keys if k not in payload]\n"
            "        if missing:\n"
            "            raise KeyError(f'stage input missing {missing}')\n"
            "        return {k: payload[k] for k in required_keys}\n"
            "    return transfer\n"
            "talker_input = make_transfer(['hidden', 'text'])\n"
            "print(talker_input({'hidden': [1, 2, 3], 'text': 'hi', 'extra': 'ignored'}))"
        ),
        md("## Exercise\n\n"
           "Call `talker_input` with a payload missing `hidden`. What error surfaces, and why is failing "
           "loudly at the edge better than passing a bad tensor downstream?"),
        code(
            "# Solution\n"
            "try:\n"
            "    talker_input({'text': 'hi'})\n"
            "except KeyError as e:\n"
            "    print('caught at the edge:', e)\n"
            "print('Failing here localizes the bug to the transfer contract, not a stage forward pass.')"
        ),
        raw(
            "# Linux GPU lab (not run here) — once your deploy YAML exists, it loads by name:\n"
            "vllm serve my-org/my-omni-model --omni --port 8091\n"
            "# vLLM-Omni reads vllm_omni/deploy/<model>.yaml to build the stage graph."
        ),
        md("## Source lab\n\nOpen `vllm_omni/deploy/qwen3_omni_moe.yaml` and one diffusion config. Map every "
           "field above to the real schema, and find where model code registers its stage classes."),
    ]


# ----------------------------------------------------------------------------- 12
def nb12():
    return [
        header("Deployment and graduation", [
            "Move from a Mac client to a Linux serving host",
            "Read a real multi-stage launch end to end",
            "Leave with a checklist for going to production",
        ]),
        md("## From Mac to Linux\n\n"
           "You built every mental model on your Mac with simulations. Production is the same graph on real "
           "hardware: install vLLM-Omni on a Linux + NVIDIA host, launch the stages, and point your client "
           "at the HTTP server. Your Mac becomes the *client*."),
        code(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Deployment:\n"
            "    model: str\n"
            "    version: str\n"
            "    port: int = 8091\n"
            "    def commands(self):\n"
            "        return [\n"
            "            f'uv pip install vllm=={self.version} --torch-backend=auto',\n"
            "            f'git clone --branch v{self.version} https://github.com/vllm-project/vllm-omni.git',\n"
            "            f'vllm serve {self.model} --omni --port {self.port}',\n"
            "        ]\n"
            "d = Deployment('Qwen/Qwen3-Omni-30B-A3B-Instruct', '0.24.1')\n"
            "for c in d.commands():\n"
            "    print('$', c)"
        ),
        md("## Production checklist\n\n"
           "- pin the model + vLLM-Omni version\n"
           "- scale the **bottleneck stage** first (the Talker for speech; the DiT for images)\n"
           "- keep stages on one node while payloads are large (shared-memory connector); go multi-node "
           "with Mooncake only when you must\n"
           "- benchmark with warmup dropped and the tail (p99) reported\n"
           "- watch per-stage capacity, not just aggregate throughput"),
        code(
            "checklist = ['pin versions', 'scale bottleneck', 'right connector',\n"
            "             'warmup + p99', 'per-stage metrics']\n"
            "for i, item in enumerate(checklist, 1):\n"
            "    print(f'{i}. {item}')"
        ),
        md("## Exercise\n\n"
           "You are deploying Qwen3-Omni for live voice. Which stage do you give the most replicas, and "
           "which connector do you start with? Justify from what you learned in nb07 and nb08."),
        code(
            "# Solution\n"
            "print('Most replicas -> Talker (it decodes ~3.6x more tokens; it is the bottleneck).')\n"
            "print('Connector -> start single-node shared-memory (5.49/0.53 ms); go Mooncake only if you')\n"
            "print('must span nodes for accelerators.')"
        ),
        raw(
            "# Linux GPU lab (not run here) — full disaggregated launch, one process per stage:\n"
            "CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --port 8091 --stage-id 0 --omni-master-address 127.0.0.1 --omni-master-port 26000\n"
            "CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --stage-id 1 --headless --omni-master-address 127.0.0.1 --omni-master-port 26000\n"
            "CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \\\n"
            "    --stage-id 2 --headless --omni-master-address 127.0.0.1 --omni-master-port 26000"
        ),
        md("## Graduation\n\n"
           "You can now read the vLLM-Omni architecture, reason about its performance, and deploy it. Go "
           "build something. Primary sources: the "
           "[docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/), the "
           "[repo](https://github.com/vllm-project/vllm-omni), and the "
           "[paper](https://arxiv.org/abs/2602.02204)."),
    ]


NOTEBOOKS = {
    "00_course_orientation.ipynb": nb00,
    "01_from_llm_to_omni.ipynb": nb01,
    "02_vllm_foundations.ipynb": nb02,
    "03_autoregressive_stages.ipynb": nb03,
    "04_diffusion_stages.ipynb": nb04,
    "05_stage_graphs.ipynb": nb05,
    "06_request_lifecycle.ipynb": nb06,
    "07_connectors_and_disaggregation.ipynb": nb07,
    "08_scheduling_batching_scaling.ipynb": nb08,
    "09_apis_and_streaming.ipynb": nb09,
    "10_observability_and_benchmarking.ipynb": nb10,
    "11_adding_a_model.ipynb": nb11,
    "12_deployment_and_graduation.ipynb": nb12,
}


def main():
    for name, fn in NOTEBOOKS.items():
        build(name, fn())


if __name__ == "__main__":
    main()
