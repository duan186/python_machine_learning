import queue
import threading
import time

exitFlag = 0


# 自定义多线程类
class myThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("开启线程：" + self.name)
        process_data(self.name, self.q)
        print("退出线程：" + self.name)


# 公共方法
def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print("%s processing %s" % (threadName, data))
        else:
            queueLock.release()
        time.sleep(1)


# 变量定义区
# 线程名列表
threadList = ["Thread-1", "Thread-2", "Thread-3"]
# 执行优先级标记列表
nameList = ["One", "Two", "Three", "Four", "Five"]
# 线程同步锁
queueLock = threading.Lock()
# 队列列表
workQueue = queue.Queue(10)
# 线程列表
threads = []
threadID = 1

# 创建新线程创建并运行了3个
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

# 填充队列，填充5个
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

# 等待队列清空
while not workQueue.empty():
    pass

# 通知线程是时候退出
exitFlag = 1

# 等待所有线程完成
for t in threads:
    t.join()
print("退出主线程")