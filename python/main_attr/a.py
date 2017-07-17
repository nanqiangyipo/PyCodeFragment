import b
import sys
print('in module a',__name__,__file__,sep='--》')


def main():
    print("running a.main")

b.run()
if __name__=='__main__':
    pass
    # b.run()
