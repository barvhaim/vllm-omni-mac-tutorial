from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Stage:
    name: str
    kind: str
    fn: Callable[[Any], Any]
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

def build_voice_pipeline() -> StageGraph:
    thinker=Stage("thinker","AR",lambda prompt:{"text":f"Answer: {prompt}","hidden":[1,2,3]})
    talker=Stage("talker","AR",lambda x:{"text":x["text"],"audio_codes":[7,4,7,4]})
    vocoder=Stage("vocoder","codec",lambda x:{"text":x["text"],"waveform_samples":len(x["audio_codes"])*2400})
    return StageGraph([thinker,talker,vocoder])
