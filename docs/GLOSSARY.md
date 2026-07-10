# Glossary

- **AR (autoregressive):** generation where each token depends on prior tokens.
- **Any-to-any:** a model family handling combinations of text, image, audio, and video inputs and outputs.
- **CFG:** classifier-free guidance, combining conditioned and unconditioned diffusion predictions.
- **Connector:** transport abstraction moving intermediate data between stages.
- **Continuous batching:** dynamic scheduling of active requests rather than fixed static batches.
- **DiT:** Diffusion Transformer used in iterative latent generation.
- **Disaggregation:** placing and scaling pipeline stages independently.
- **JCT:** job completion time from request arrival to final output.
- **KV cache:** stored attention keys/values from already processed tokens.
- **OmniRequestOutput:** public envelope for heterogeneous request results.
- **PagedAttention:** block-oriented KV-cache memory management used by vLLM.
- **Prefill:** initial prompt processing that creates attention state.
- **Stage graph:** nodes are model stages; edges transform and route intermediate data.
- **Talker:** AR component that may generate audio codec tokens.
- **Thinker:** comprehension/reasoning AR component in Thinker-Talker models.
- **VAE:** encoder/decoder commonly mapping images or videos to/from latent space.
- **Vocoder/Code2Wav:** component reconstructing waveform audio from intermediate representations.
