from functools import wraps ##导入这个包

def show_me(func):
    @wraps(func)
    def wrapper():
        print ("It is in wrapper.")
        func()
    return wrapper

@show_me
def func1():
    '''
    this is func1
    '''
    print ("running func1")

if __name__ == '__main__':
    func1()
    print (func1.__name__)
    print (func1.__doc__)