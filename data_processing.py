import string

def load_file(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        text = ' '.join([line.strip() for line in lines])

        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.replace('“', '')
        text = text.replace('”', '')
        text = text.replace('—', '')
    return text

def tokenize_chars(text):
    chars = sorted(set(text))
    vocab_size = len(chars)

    strToInt = {c: i for i, c, in enumerate(chars)}
    intToStr = {i: c for i, c in enumerate(chars)}

    encode = lambda s: [strToInt[c] for c in s]
    decode = lambda l: ' '.join([intToStr[i] for i in l])
    return encode, decode, vocab_size

def tokenize_words(text):
    words = set(text.split(" "))
    vocab_size = len(words)

    strToInt = {w: i for i, w, in enumerate(words)}
    intToStr = {i: w for i, w in enumerate(words)}

    encode = lambda s: [strToInt[w] for w in s]
    decode = lambda l: ''.join([intToStr[i] for i in l])
    return encode, decode, vocab_size
