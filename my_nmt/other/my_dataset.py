from .word_dictionary import WordDictionary
from torch.utils.data import Dataset
from copy import copy

class MyDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, special_token, src_w2id = None, src_id2w = None, tgt_w2id = None, tgt_id2w = None) -> None:
        super().__init__()
        
        self.src_path = src_path
        self.tgt_path = tgt_path

        #辞書を作成
        self.src_dict = WordDictionary(special_token)
        self.tgt_dict = WordDictionary(special_token)

        if src_id2w == None:
            self.src_w2id, self.src_id2w = self.src_dict.create_get_dict(src_path)
            self.tgt_w2id, self.tgt_id2w = self.tgt_dict.create_get_dict(tgt_path)
        else:
            self.src_w2id = src_w2id.copy()
            self.src_id2w = src_id2w.copy()
            self.tgt_w2id = tgt_w2id.copy()
            self.tgt_id2w = tgt_id2w.copy()
        
        self.special_token = special_token.copy()
        
        self.src_idlines = self.wordlines_2_idlines(src_path, self.src_w2id)
        self.tgt_idlines = self.wordlines_2_idlines(tgt_path, self.tgt_w2id)
    
    def __len__(self):
        return len(self.src_idlines)

    # indexに対応した文章ペア (src, tgt)
    def __getitem__(self, index):
        return self.add_bos_eos(self.src_idlines[index]), self.add_bos_eos(self.tgt_idlines[index])

    # 語彙サイズ (src,tgt)
    def get_vocab_size(self):
        return len(self.src_w2id), len(self.tgt_w2id)
    
    def get_tgt_id2w(self):
        return self.tgt_id2w
    
    def get_src_id2w(self):
        return self.src_id2w
    
    def get_dicts(self):
        return self.src_w2id, self.src_id2w, self.tgt_w2id, self.tgt_id2w
    
    def wordlines_2_idlines(self, path: str, w2id: dict) -> list:
        result = []

        with open(path, "r") as f:
            tmp_doc = f.read().splitlines()
            for line in tmp_doc:
                tmp_list = []
                for word in line.split(" "):
                    l_word = word.lower()
                    if l_word in w2id:
                        tmp_list.append(w2id[l_word])
                    else:
                        tmp_list.append(self.special_token["<unk>"])
                result.append(tmp_list)
        return result

    def add_bos_eos(self, ids):
        tmp_list = copy(ids)
        tmp_list.insert(0, self.special_token["<bos>"])
        tmp_list.append(self.special_token["<eos>"])
        return tmp_list
    
    def get_all_tgt(self):
        result = []

        with open(self.tgt_path, "r") as f:
            tmp_doc = f.read().splitlines()
            for line in tmp_doc:
                tmp_list = []
                for word in line.split(" "):
                    l_word = word.lower()
                    tmp_list.append(l_word)
                result.append(tmp_list)
        return result