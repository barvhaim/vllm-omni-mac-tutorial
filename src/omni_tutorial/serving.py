from dataclasses import dataclass
import heapq

@dataclass
class StageSpec:
    name: str
    service_time: float
    replicas: int = 1

class PipelineSimulator:
    """Deterministic tandem-queue simulator for educational experiments."""
    def __init__(self, stages: list[StageSpec]):
        if any(s.service_time <= 0 or s.replicas < 1 for s in stages):
            raise ValueError("service_time must be positive and replicas >= 1")
        self.stages=stages
    def run(self, arrivals: list[float]) -> dict:
        completions=list(arrivals); stage_stats=[]
        for stage in self.stages:
            workers=[0.0]*stage.replicas; heapq.heapify(workers); next_completions=[]; waits=[]
            for arrival in completions:
                free=heapq.heappop(workers); start=max(arrival,free)
                waits.append(start-arrival); finish=start+stage.service_time
                heapq.heappush(workers,finish); next_completions.append(finish)
            completions=next_completions
            stage_stats.append({"stage":stage.name,"mean_wait":sum(waits)/len(waits),"max_wait":max(waits),"replicas":stage.replicas})
        return {"completions":completions,"makespan":max(completions)-min(arrivals),"stage_stats":stage_stats}
