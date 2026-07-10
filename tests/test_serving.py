from omni_tutorial.serving import PipelineSimulator, StageSpec

def test_scaling_bottleneck_reduces_makespan():
    arrivals=[i*0.5 for i in range(20)]
    slow=PipelineSimulator([StageSpec("thinker",1),StageSpec("talker",3,1),StageSpec("vocoder",.5)]).run(arrivals)
    scaled=PipelineSimulator([StageSpec("thinker",1),StageSpec("talker",3,3),StageSpec("vocoder",.5)]).run(arrivals)
    assert scaled["makespan"] < slow["makespan"]
