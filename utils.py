import re

import paddle
from paddlenlp.datasets import MapDataset

hash_map = {'P': '收件人', 'T': '电话号码', 'A1': '省', 'A2': '市', 'A3': '区/县', 'A4': '详细地址', 'O': '其他'}


def load_testdata(arg):
    if arg[0] == '-a':
        with open(arg[1], 'r', encoding='utf-8') as fp:
            return fp.readlines()
    elif arg[0] == '-b':
        return arg[1]


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab


def encode(arg, data):
    def encode_file(data):
        for line in data:
            words = [line[i] for i in range(len(line.strip('\n')))]
            labels = ['O' for i in range(len(line))]
            yield words, labels

    def read_line(data):
        words = [data[i] for i in range(len(data.strip('\n')))]
        labels = ['O' for i in range(len(data))]
        yield words, labels

    if arg[0] == '-a':
        return MapDataset(list(encode_file(data)))
    elif arg[0] == '-b':
        return MapDataset(list(read_line(data)))


def convert_example(example, tokenizer, label_vocab):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']
    tokenized_input['labels'] = [label_vocab[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))


def predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = decode(ds, pred_list, len_list, label_vocab)
    return preds


def decode(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))
    res = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(hash_map.get(t.split('-')[0]))
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        res.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return res


def run_re(arg, data):
    provinces = ['北京', '天津', '河北', '山西', '内蒙古自治区', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西',
                 '山东', '河南', '湖北', '湖南', '广东', '广西壮族自治区', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏自治区', '西藏', '陕西', '甘肃',
                 '青海', '宁夏回族自治区', '宁夏', '新疆维吾尔自治区', '新疆', '台湾', '香港特别行政区', '香港', '澳门特别行政区', '澳门']
    res = []
    for line in data:
        sent_out = []
        tags_out = []

        phone = re.search(r'1[3-9]\d{9}', line)
        if phone:
            line = re.sub(r'1[3-9]\d{9}', " ", line)
            sent_out.append(phone.group())
            tags_out.append(hash_map.get('T'))

        for province in provinces:
            pro = re.search(r'{}省?'.format(province), line)
            if pro:
                line = re.sub(r'{}省?'.format(province), ' ', line)
                sent_out.append(pro.group())
                tags_out.append(hash_map.get('A1'))
                break

        city = re.search(r'\s.*市', line)
        if city:
            line = re.sub(r'\s.*市', ' ', line)
            sent_out.append(city.group().strip(" \n"))
            tags_out.append(hash_map.get('A2'))

        district = re.search(r'\s.*[县区]', line)
        if district:
            line = re.sub(r'\s.*[县区]', ' ', line)
            sent_out.append(district.group().strip(" \n"))
            tags_out.append(hash_map.get('A3'))

        address = re.search(r'\s.*[号米路心层场街栋楼道厦镇内元东南西北汇]', line)
        if address:
            line = re.sub(r'\s.*[号米路心层场街栋楼道厦镇内元东南西北汇]', ' ', line)
            sent_out.append(address.group().strip(" \n"))
            tags_out.append(hash_map.get('A4'))

        name = re.search(r'[\u4e00-\u9fa5]{2,10}', line)
        if name:
            line = re.sub(r'[\u4e00-\u9fa5]{2,10}', ' ', line)
            sent_out.append(name.group().strip(" \n"))
            tags_out.append(hash_map.get('P'))

        res.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return res
