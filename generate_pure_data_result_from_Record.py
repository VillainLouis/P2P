import os
import logging

RECORD_PATH = '/data/jliu/P2P/Results'

def record2txt(record_path):
    # Generating .txt file
    for maindir, subdir, filelist in os.walk(record_path):
        if 'server' in maindir:
            server_log_path = None
            for _, _, fl in os.walk(maindir):
                # print("ff", ff[0])
                server_log_path = maindir + '/' + fl[0]
                #print(maindir + fl[0])
                # Init logger
            filename = server_log_path.split('.')[0] + '.txt' # 与记录同名的.txt文件
            if not os.path.exists(filename): # 只有记录文件不存在的时候才进行操作
                logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
                logger.setLevel(logging.INFO)
                fileHandler = logging.FileHandler(filename=filename)
                formatter = logging.Formatter("%(message)s")
                fileHandler.setFormatter(formatter)
                logger.addHandler(fileHandler)

                fp = open(server_log_path, encoding='utf-8')
                lines = fp.readlines()
                for line in lines:
                    if 'Epoch' in line:
                        logger.info("{} {}".format(float(line.split(',')[1].split()[2]), line.split(',')[2].split()[2]))

if __name__ == "__main__":
    record2txt(RECORD_PATH)
