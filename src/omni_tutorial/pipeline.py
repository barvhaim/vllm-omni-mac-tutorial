from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Stage:
    name: str
    kind: str
    fn: Callable[[Any], Any]
    # Relative time one request spends in this stage. Defaults to 1.0 so existing
    # callers are unaffected; notebook 08 uses it to feed the PipelineSimulator and
    # show the Talker as the bottleneck (it decodes far more tokens than the Thinker).
    service_time: float = 1.0
    def run(self, value: Any) -> Any:
        return self.fn(value)

@dataclass
class Connector:
    name: str = "in-process"
    transfers: int = 0
    def transfer(self, value: Any) -> Any:
        self.transfers += 1
        return value

class StageGraph:
    def __init__(self, stages: list[Stage], connector: Connector | None = None):
        if not stages:
            raise ValueError("a stage graph needs at least one stage")
        self.stages = stages
        self.connector = connector or Connector()
    def run(self, value: Any) -> tuple[Any, list[dict[str, Any]]]:
        trace=[]
        for index, stage in enumerate(self.stages):
            before=value
            value=stage.run(value)
            trace.append({"stage":stage.name,"kind":stage.kind,"input":before,"output":value})
            if index < len(self.stages)-1:
                value=self.connector.transfer(value)
        return value, trace
    def stage_specs(self, replicas: dict[str, int] | None = None) -> list["StageSpec"]:
        """Project this graph onto serving.StageSpec objects for the simulator.

        Each stage's `service_time` becomes the simulator's service time; pass
        `replicas` to scale specific stages (e.g. {"talker": 3}).
        """
        from .serving import StageSpec
        replicas = replicas or {}
        return [StageSpec(s.name, s.service_time, replicas.get(s.name, 1)) for s in self.stages]

def build_voice_pipeline() -> StageGraph:
    # service_time reflects the paper's finding that the Talker dominates latency:
    # it runs many more decode iterations (~545 audio vs ~150 text tokens) than the
    # Thinker, while Code2Wav is cheap. The fn outputs are unchanged.
    thinker=Stage("thinker","AR",lambda prompt:{"text":f"Answer: {prompt}","hidden":[1,2,3]},service_time=1.0)
    talker=Stage("talker","AR",lambda x:{"text":x["text"],"audio_codes":[7,4,7,4]},service_time=3.5)
    vocoder=Stage("vocoder","codec",lambda x:{"text":x["text"],"waveform_samples":len(x["audio_codes"])*2400},service_time=0.5)
    return StageGraph([thinker,talker,vocoder])
