"""
实验一部分命令行解析功能
"""
import argparse


if __name__=='__main__':
    # 相当于生成一个用于解析命令行的类
    parse = argparse.ArgumentParser(prog='czmcommd',description='命令行工具描述：实验怎么用的')

    # 参数是一个列表值
    parse.add_argument('-n', type=int ,nargs='+')

    # 参数是一个字符串
    parse.add_argument('--b')

    re = parse.parse_args()
    # re = parse.parse_args('-n 124 3 4 --b asdf'.split())

    print(re)
    print(re.n)

# 测试语句
# python commandline.py -n 124 3 4 --b asdf
# 或者
# re = parse.parse_args('-n 124 3 4 --b asdf'.split())