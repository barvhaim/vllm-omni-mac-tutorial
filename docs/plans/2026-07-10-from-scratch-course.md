# vLLM-Omni From-Scratch Course Implementation Plan

> **For Hermes:** Implement and execute the entire curriculum before publishing.

**Goal:** Turn the repository into a complete beginner-to-practical vLLM-Omni course that runs on a Mac.

**Architecture:** Thirteen notebooks progress from inference fundamentals to AR and diffusion stages, stage graphs, orchestration, connectors, scaling, APIs, benchmarking, model integration, and Linux deployment. Lightweight tested simulations run locally; clearly marked source and Linux GPU labs connect them to the official runtime.

**Tech Stack:** Python 3.11, Jupyter, NumPy, Matplotlib, pytest, nbclient, macOS GitHub Actions.

---

1. Replace the short six-notebook overview with a thirteen-module curriculum.
2. Correctly distinguish regular vLLM-Metal support from full vLLM-Omni support.
3. Add source-reading labs, checkpoints, exercises, deployment guidance, and a glossary.
4. Validate curriculum structure through automated tests.
5. Execute all notebooks headlessly and run all tests.
6. Push to GitHub and require the macOS CI workflow to pass.
