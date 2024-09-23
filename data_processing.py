def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def tokenize(text):
    chars = sorted(set(text))
    vocab_size = len(chars)

    strToInt = {c: i for i, c, in enumerate(chars)}
    intToStr = {i: c for i, c in enumerate(chars)}

    encode = lambda s: [strToInt[c] for c in s]
    decode = lambda l: ''.join([intToStr[i] for i in l])
    return encode, decode, vocab_size
