import os,sys
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hot_module
import time
import re
import inspect


if __name__=='__main__':
    rule_pattern = re.compile('^action_.*')
    while True:
        # 重新加载模块
        importlib.reload(hot_module)
        # 存放
        rule_method_name = []
        #
        dahuang = hot_module.dog("huang")
        for item in hot_module.dog.__dict__.items():
            # print(item[0],type(item[1]))

            if inspect.isfunction(item[1]) and rule_pattern.match(item[0]):
                rule_method_name.append(item[0])

        for name in rule_method_name :
            rule_method = getattr(dahuang,name)
            rule_method()

        # print(hot_module.dog.__dict__)

        # d = dog()


        time.sleep(5)
