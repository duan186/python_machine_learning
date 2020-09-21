import threading
import time

class myThread(threading.Thread):
    def __init__(self,threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print("开始线程：" + self.name)
        # 获取锁，用于线程同步
        threadLock.acquire()
        print_time(self.name, self.counter, 3)
        print("退出线程：" + self.name)
        # 释放锁，开启下一个线程
        threadLock.release()


def print_time(threadName,delay,counter):
        while counter:
            time.sleep(delay)
            print("%s : %s" % (threadName, time.ctime(time.time())))
            counter -= 1

threadLock = threading.Lock()
threads = []
# 创建新线程
thread1 = myThread(1, "Tread - 1", 1)
thread2 = myThread(2, "Tread - 2", 2)

# 开启新线程
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print("退出主线程")

