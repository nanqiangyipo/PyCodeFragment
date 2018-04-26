import logging
import logging.handlers
from pathlib import Path
import os



def init_log(level = logging.INFO,name='main'):
    parent_path = Path(os.path.dirname(os.path.abspath(__file__)))
    log_path = parent_path.joinpath('logs',f'{name}.log')
    log_path.parent.mkdir(exist_ok=True, parents=True)
    rf_handler = logging.handlers.RotatingFileHandler(log_path,maxBytes=1024,backupCount=3)
    log_format = "%(levelname)s %(asctime)s %(pathname)s %(funcName)s %(lineno)d > %(message)s"
    logging.basicConfig(format=log_format,level=level,handlers=[rf_handler])


if __name__ == '__main__':
    init_log(name='test.log')
    while True:
        a = input("回车一下：")
        if a !='':
            print(a)
            logging.info(a)
        else:
            logging.info(msg="我是一条日志")
