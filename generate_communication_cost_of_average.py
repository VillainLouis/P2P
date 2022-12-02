import os
import logging
# import pdb

RECORD_PATH = '/data/jliu/P2P/Results'

def record2txt(record_path):
    # Generating .txt file
    for maindir, subdir, filelist in os.walk(record_path):
        if 'clients' in maindir: # 所有记录
            print(maindir) 
            clients_log_path = None
            for _, _, fl in os.walk(maindir): # 记录下所有client记录
                # print(fl)
                _comm_cost = list() # 用一个list累加所有client所有epoch的通信量
    
                _comm_time = list()
                _first = True 
                _client_num = 0
                for log in fl: # 遍历所有的client记录文件
                    _client_num += 1
                    # print(log)
                    # pdb.set_trace()
                    clients_log_path = maindir + '/' + log # 每个client的记录文件的绝对路径
                    if _first:
                        _first = False
                        with open(clients_log_path, encoding='utf-8') as fp:
                            lines = fp.readlines()
                            for line in lines: # 将每个周期的通信量都记下来
                                if 'Communication Traffic' in line:
                                    _comm_cost.append(float(line.split()[10][:-3]))
                                if 'Communication Time' in line:
                                    _comm_time.append(float(line.split()[10][:-2]))
                    else:
                        with open(clients_log_path, encoding='utf-8') as fp:
                            lines = fp.readlines()
                            c_idx = 0 # 记录epoch
                            t_idx = 0
                            for line in lines: # 将每个周期的通信量都记下来
                                if 'Communication Traffic' in line:
                                    _comm_cost[c_idx] += float(line.split()[10][:-3])
                                    # print(c_idx)
                                    c_idx += 1
                                if 'Communication Time' in line:
                                    _comm_time[t_idx] += float(line.split()[10][:-2])
                                    t_idx += 1
                _comm_cost = [x / _client_num for x in _comm_cost]
                _comm_time = [x / _client_num for x in _comm_time]

                
                #print(maindir + fl[0])
                # Init logger
                filename = maindir + '_communication' + '.txt' # 与记录同名的.txt文件
                os.system('rm {}'.format(filename))
                # if not os.path.exists(filename): # 只有记录文件不存在的时候才进行操作
                logger = logging.getLogger(filename)
                logger.setLevel(logging.INFO)
                fileHandler = logging.FileHandler(filename=filename)
                formatter = logging.Formatter("%(message)s")
                fileHandler.setFormatter(formatter)
                logger.addHandler(fileHandler)
                
                _cost = 0
                _time = 0
                for x, y in zip(_comm_cost, _comm_time):
                    _cost += x
                    _time += y
                    logger.info("{} {}".format(_cost, _time))
            #     fp = open(clients_log_path, encoding='utf-8')
            #     lines = fp.readlines()
            #     for line in lines:
            #         if 'Epoch' in line:
            #             logger.info("{} {}".format(float(line.split(',')[1].split()[2]), line.split(',')[2].split()[2]))

if __name__ == "__main__":
    record2txt(RECORD_PATH)
