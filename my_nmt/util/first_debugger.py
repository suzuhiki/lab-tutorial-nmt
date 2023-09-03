import sys
from collections import defaultdict

# 1回目の処理のみをデバッグする
class FirstDebugger:
    # クラス変数(インスタンス間で共有)
    debug_dict = defaultdict(int)
    first_stop = False
    
    def debug(self, key: str, info: str) -> None:
        if self.debug_dict[key] == 0:
            self.debug_dict[key] += 1
            print("key:{}\ninfo:{}".format(key, info))
        else:
            if self.first_stop:
                print("FirstDebugger: key({})が2回入力されたため終了します".format(key))
                sys.exit()