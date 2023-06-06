import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from utils import convert_example, evaluate, load_dict, load_dataset
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import ChunkEvaluator

train_ds, dev_ds = load_dataset(datafiles=('./express_ner/train.txt', './express_ner/dev.txt'))

label_vocab = load_dict('./conf/tag.dic')

MODEL_NAME = "ernie-1.0"
tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)

ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)

train_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_size=200,
    return_list=True,
    collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_size=200,
    return_list=True,
    collate_fn=batchify_fn)

model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))

metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

step = 0
for epoch in range(10):
    model.train()
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step += 1
        print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    evaluate(model, metric, dev_loader)

model.save_pretrained('./checkpoint')
tokenizer.save_pretrained('./checkpoint')
