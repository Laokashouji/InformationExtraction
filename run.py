import sys
import paddle
from getopt import getopt
from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer
from functools import partial
from utils import predict, load_dict, load_testdata, convert_example, run_re, encode
from paddlenlp.data import Stack, Tuple, Pad


def run_model(test_ds):
    label_vocab = load_dict('./conf/tag.dic')
    tokenizer = ErnieTokenizer.from_pretrained("./checkpoint")
    trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)
    test_ds.map(trans_func)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=-1)  # labels
    ): fn(samples)
    test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=200,
        return_list=True,
        collate_fn=batchify_fn)
    model = ErnieForTokenClassification.from_pretrained("./checkpoint/")
    return predict(model, test_loader, test_ds, label_vocab)


opts, args = getopt(sys.argv[1:], 'a:b:r')
testdata = load_testdata(opts[0])

if len(opts) > 1 and opts[1][0] == '-r':
    for result in run_re(opts[0], testdata):
        print(result)
else:
    for result in run_model(encode(opts[0], testdata)):
        print(result)
