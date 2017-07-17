import sys
print('in module b',__name__,__file__,sep='--》')

def run():
    print("running b")
    main = sys.modules['__main__'].main
    main()
    print("b running finished")