import sys
from collections import defaultdict

# 1回目の処理のみをデバッグする
class FirstDebugger:
    # クラス変数(インスタンス間で共有)
    debug_dict = defaultdict(int)
    count = 1
    
    def debug(self, key: str, info: str) -> None:
        if self.debug_dict[key] == 0:
            self.debug_dict[key] += 1
            print("key:{}\ninfo:{}".format(key, info))
        elif self.debug_dict[key] >= self.count:
            print("FirstDebugger: key({})が2回入力されたため終了します".format(key))
            sys.exit()