# GLASSBOX — Project Brief

An interactive transformer interpretability tool. The goal is to *see* what a language model does internally as it runs, by instrumenting and visualizing its data flow.

## Why I'm building this

I want to do something fun with the models themselves.

I'm a visual thinker — I understand systems through data flow and by visualizing them. I've spent years building AI agents and AI platforms, so I'm fluent in LLMs as a user and a systems builder, but I have almost no knowledge of how models work *internally*. I don't have a formal ML-math background and don't want to acquire one just to do this. I want something fun and interesting that teaches me model internals the way I learn everything else: by plumbing data through a system and watching it move.

## The core insight

A transformer literally *is* a data-flow system, which is exactly the kind of thing my brain is built for.

There's a "highway" running straight through the model — the **residual stream**. Every token gets a vector that travels up through the layers, and each layer *reads* from that vector, computes something, and *writes back* to it. Tokens pass information to each other through **attention**. It's the same mental model as an agent graph where data flows between nodes — just one level down.

The breakthrough trick that makes this visual is the **logit lens**: you can take a snapshot of a token's vector at *any* layer and decode it back into "what word would the model predict right now?" That lets you watch the model make up its mind, layer by layer.

For example, given the prompt *"The Eiffel Tower is in the city of ___"*, the prediction isn't sitting in the model waiting — it gets *assembled* as the vector flows upward:

- Early layers: generic guesses (`the`, `world`) — surface grammar only, no reasoning yet.
- Middle layers: associations form (`lights`, then `France`) — it knows the landmark and the country, but not the city.
- Later layers: `Paris` emerges as the top guess and confidence sharpens toward the output.

The fact "Paris" doesn't exist at layer 1. It materializes out of noise as the data flows. The first time you run this on a real model and watch that happen, it's a small jolt — and that short feedback loop is the whole appeal of the field.

## The field: mechanistic interpretability

This belongs to a field called **mechanistic interpretability** ("mech interp"). The entire premise is reverse-engineering what a trained model is actually doing inside, mostly by instrumenting and visualizing the data flow. The appeal — as the tool authors themselves describe it — is the *extremely short feedback loops*: poke the model, see something change, immediately.

Why it fits me specifically: I don't need to understand how models are *trained* (the calculus-heavy part). I only need to watch what flows through one as it runs. That's a systems-and-data-flow problem — my home turf.

## The project

**Build an interactive "model inspector" — the tool I wish existed while learning.**

Feed in a sentence, and a web UI shows, live:

- **Logit lens** — the model's top predicted next-token, decoded at each layer.
- **Attention patterns** — which tokens are looking at which.
- **Layer ablation** — how the prediction shifts when a layer is removed.

It sits exactly on my existing stack:

- **Python backend** hooks into a small open model and extracts activations. A FastAPI endpoint takes a prompt and returns the per-layer data. This is home turf — it's just a data pipeline, except the data comes out of a neural net instead of Postgres.
- **React frontend** renders the data flow. My visual instinct pays off here — interpretability tools are rarely good-looking, so the bar is low.
- **Model internals** are the genuinely new part — learned by plumbing the data through, not by deriving equations.

## Tooling

The field has two main entry points, and they split cleanly:

- **TransformerLens** — loads an open-source model like GPT-2 and exposes its internal activations. You can cache any internal activation and add functions to edit, remove, or replace activations as the model runs. It's the classic starting point, built around GPT-2-small, and the whole tutorial ecosystem (ARENA, Neel Nanda's videos) is written for it. **Start here.**
- **nnsight / nnterp** — works directly on HuggingFace models. Its key advantage is running interpretability experiments on models too large to run locally: it serializes the intervention, sends it to a remote facility where the model is already loaded, and returns only the requested tensors, so you never download hundreds of billions of parameters. Overkill for now, but good to know it exists when GPT-2 stops being interesting.

**Stack:** Python · TransformerLens · PyTorch · FastAPI (backend) · React (frontend). On macOS (Apple Silicon), so prefer setups that work well with MPS and flag anything CUDA-only.

## Roadmap

Roughly a weekend each. Update as it evolves.

1. **Weekend 1 — Feel the feedback loop.** `pip install transformer_lens`, load GPT-2-small, reproduce the logit-lens plot on five prompts of my choice. No web UI yet — just a notebook and matplotlib.
2. **Weekend 2 — Make it live.** Wrap it in FastAPI + a minimal React page. One input box, one logit-lens visualization that updates live.
3. **Weekend 3+ — Make it mine.** Add attention-pattern views, then layer ablation ("what breaks if I delete layer 7?"), maybe a feature browser. This is where it becomes portfolio-worthy — and it stands out precisely because my GitHub is all backend infra, and this is something nobody expects from that profile.
4. **Stretch.** Feature browsing / sparse autoencoders (SAEs), and polish it enough to open-source.
