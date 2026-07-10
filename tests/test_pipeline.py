from omni_tutorial import build_voice_pipeline

def test_voice_pipeline_and_connector_trace():
    graph=build_voice_pipeline(); output,trace=graph.run("hello")
    assert output["waveform_samples"] == 9600
    assert [x["stage"] for x in trace] == ["thinker","talker","vocoder"]
    assert graph.connector.transfers == 2
