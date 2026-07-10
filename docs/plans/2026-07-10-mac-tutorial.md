# vLLM-Omni Mac Tutorial Implementation Plan

> **For Hermes:** Implement and verify every notebook as an executable artifact.

**Goal:** Build a Mac-compatible notebook course that accurately teaches vLLM-Omni without claiming the production runtime supports macOS.

**Architecture:** Lightweight NumPy simulations expose AR, diffusion, stage-graph, connector, and queueing concepts. Shared tested Python primitives prevent notebook duplication. The final module bridges concepts to a real Linux deployment.

**Tech Stack:** Python 3.11, Jupyter, NumPy, Matplotlib, pytest, nbclient, GitHub Actions on macOS.

---

1. Scaffold package metadata, Mac bootstrap, README, and compatibility warning.
2. Implement and unit-test stage graph and connector primitives.
3. Implement and unit-test the deterministic serving simulator.
4. Add six notebooks in concept-first order.
5. Execute every notebook headlessly and run unit tests.
6. Add macOS CI, publish to GitHub, and verify the remote repository.
