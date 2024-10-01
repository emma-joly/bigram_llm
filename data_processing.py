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

def tokenize_words(text, max_words):
    wordsCount = dict()
    words = text.split(" ")
    vocab_size = len(set(words))

    for word in words:
        wordsCount[word] = wordsCount.get(word, 0) + 1

    top_words = sorted(wordsCount.items(), key=lambda x: x[1], reverse=True)

    if vocab_size > max_words:
        words = set([w[0] for w in top_words[0:max_words]])
        vocab_size = len(words)
    else:
        words = set(words)

    strToInt = {w: i for i, w, in enumerate(words)}
    intToStr = {i: w for i, w in enumerate(words)}

    encode = lambda s: [strToInt[w] for w in s if w in strToInt]
    decode = lambda l: ''.join([intToStr[i] + ' ' for i in l])
    return encode, decode, vocab_size

