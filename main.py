from paddlenlp import Taskflow

ner_fast = Taskflow("ner", mode="fast")

print(ner_fast("三亚是一个美丽的城市"))
