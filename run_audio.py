from paddlespeech.cli.asr.infer import ASRExecutor

audio = "./express_ner/test_audio.wav"
asr = ASRExecutor()
result = asr(audio_file=audio, model='conformer_online_wenetspeech')

print(result)
