class IdAttacher:
    def __init__(self) -> None:
        self.init_dict()

    def set_dict(self, word: str):
        word_tmp = word.lower()
        if word_tmp not in self.w2id_dict:
            id = len(self.w2id_dict)
            self.w2id_dict[word_tmp] = id
            self.id2w_dict[id] = word_tmp

    def init_dict(self):
        self.w2id_dict = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}
        self.id2w_dict = {0:"<PAD>", 1:"<BOS>", 2:"<EOS>", 3:"<UNK>"}

    def add_dict(self, doc):
        for sentence in doc:
            for word in sentence:
                self.set_dict(word)
    
    def word2id(self, word:str):
        if word in self.w2id_dict:
            return self.w2id_dict[word]
        else:
            return self.w2id_dict["<UNK>"]

