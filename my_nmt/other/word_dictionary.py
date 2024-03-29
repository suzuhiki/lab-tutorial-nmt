from ..util.util import get_swap_dict

class WordDictionary:
    def __init__(self, special_token) -> None:
        self.special_token = special_token.copy()
        self.w2id_dict = {}
        self.id2w_dict = {}
    
    # 辞書の初期化と構築
    def create_dict(self, src_path: str):
        self.w2id_dict = self.special_token
        self.id2w_dict = get_swap_dict(self.special_token)
        with open(src_path, "r") as f:
            tmp_doc = f.read().splitlines()
            for line in tmp_doc:
                if line == "":
                    continue
                tokens = line.split(" ")
                for word in tokens:
                    l_word = word.lower()
                    if l_word not in self.w2id_dict:
                        id = len(self.w2id_dict)
                        self.w2id_dict[l_word] = id
                        self.id2w_dict[id]=l_word

    
    def get_w2id_dict(self) -> dict:
        return self.w2id_dict
    
    def get_id2w_dict(self) -> dict:
        return self.id2w_dict

    # 辞書の作成とid2w、w2idの取得 (w2id, id2w)
    def create_get_dict(self, src_path: str) -> (dict, dict):
        self.create_dict(src_path)
        return self.get_w2id_dict(), self.get_id2w_dict()