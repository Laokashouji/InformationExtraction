from paddlespeech.cli.asr.infer import ASRExecutor
from paddlenlp import Taskflow

schema = ["时间", "出发地", "目的地", "费用"]
ie = Taskflow("information_extraction",
              schema=schema, task_path="./audio_model/")
asr = ASRExecutor()


def audio_ie(audio_path):
    asr_result = asr(audio_file=audio_path, force_yes=True)
    ie_result = ie(asr_result)
    return ie_result


audio = "./data_set/test_audio.wav"
print(audio_ie(audio))
