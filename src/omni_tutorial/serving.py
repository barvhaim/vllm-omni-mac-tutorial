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
            # effective throughput of a stage is replicas / service_time; its
            # reciprocal is the per-request time the stage can sustain.
            capacity=stage.replicas/stage.service_time
            stage_stats.append({"stage":stage.name,"mean_wait":sum(waits)/len(waits),"max_wait":max(waits),"replicas":stage.replicas,"capacity":capacity})
        # the bottleneck is the stage with the lowest sustained throughput.
        slowest=min(stage_stats,key=lambda s:s["capacity"])
        for s in stage_stats:
            s["bottleneck"]=(s["stage"]==slowest["stage"])
        return {"completions":completions,"makespan":max(completions)-min(arrivals),"stage_stats":stage_stats,"bottleneck":slowest["stage"]}


def sweep_replicas(base_stages: list[StageSpec], stage_name: str, replica_counts: list[int], arrivals: list[float]) -> list[dict]:
    """Vary one stage's replica count and report makespan + its max wait.

    Returns one row per replica count, suitable for plotting the diminishing
    returns of scaling a single stage.
    """
    rows=[]
    for n in replica_counts:
        stages=[StageSpec(s.name,s.service_time,n if s.name==stage_name else s.replicas) for s in base_stages]
        result=PipelineSimulator(stages).run(arrivals)
        stage_wait=next(s["max_wait"] for s in result["stage_stats"] if s["stage"]==stage_name)
        rows.append({"replicas":n,"makespan":result["makespan"],"stage_max_wait":stage_wait,"bottleneck":result["bottleneck"]})
    return rows
