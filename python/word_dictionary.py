class WordDictionary:
    def __init__(self) -> None:
        self.w2id_dict_ja = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
        self.id2w_dict_ja = {0:"<pad>", 1:"<bos>", 2:"<eos>", 3:"<unk>"}
        self.w2id_dict_en = {"<pad>":0, "<box>":1, "<eos>":2, "<unk>":3}
        self.id2w_dict_en = {0:"<pad>", 1:"<bos>", 2:"<eos>", 3:"<unk>"}
        
        self.SRC_PATH = "/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/"
    
    def create_dict(self):
        # JA
        self.w2id_dict_ja = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
        self.id2w_dict_ja = {0:"<pad>", 1:"<bos>", 2:"<eos>", 3:"<unk>"}
        
        with open(self.SRC_PATH + "train-1.short.ja", "r") as f:
            tmp_doc = f.read().splitlines()
            for line in tmp_doc:
                tmp_line = line.split(" ")
                for word in tmp_line:
                    l_word = word.lower()
                    if l_word not in self.w2id_dict_ja:
                        id = len(self.w2id_dict_ja)
                        self.w2id_dict_ja[l_word] = id
                        self.id2w_dict_ja[id] = l_word
        
        # EN
        self.w2id_dict_en = {"<pad>":0, "<box>":1, "<eos>":2, "<unk>":3}
        self.id2w_dict_en = {0:"<pad>", 1:"<bos>", 2:"<eos>", 3:"<unk>"}
        
        with open(self.SRC_PATH + "train-1.short.en", "r") as f:
            tmp_doc = f.read().splitlines()
            for line in tmp_doc:
                tmp_line = line.split(" ")
                for word in tmp_line:
                    l_word = word.lower()
                    if l_word not in self.w2id_dict_en:
                        id = len(self.w2id_dict_en)
                        self.w2id_dict_en[l_word] = id
                        self.id2w_dict_en[id] = l_word
    
    def get_id(self, data_id: str, lang: str):
        if data_id in ["train-1.short", "dev", "test"] and lang in ["en", "ja"]:
            w2id_dict = {}
            if lang == "ja":
                w2id_dict = self.w2id_dict_ja
            else:
                w2id_dict = self.w2id_dict_en
            
            path = self.SRC_PATH + data_id + "." + lang
            result_list = []
            with open(path, "r") as f:
                tmp_doc = f.read().splitlines()
                for line in tmp_doc:
                    tmp_list = []
                    for word in line.split(" "):
                        l_word = word.lower()
                        if l_word in w2id_dict:
                            tmp_list.append(w2id_dict[l_word])
                        else:
                            tmp_list.append(w2id_dict["<unk>"])
                    result_list.append(tmp_list)
            return result_list
    
    def get_dict(self, lang: str, type: str):
        if lang == "en" and type == "w2id":
            return self.w2id_dict_en
        elif lang == "en" and type == "id2w":
            return self.id2w_dict_en
        elif lang == "ja" and type == "w2id":
            return self.w2id_dict_ja
        elif lang == "ja" and type == "id2w":
            return self.id2w_dict_ja