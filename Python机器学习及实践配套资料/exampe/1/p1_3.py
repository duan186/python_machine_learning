# iter(IterableObject)
ita = iter([1, 2, 3])
print(type(ita)) 
print(next(ita))
print(next(ita))
print(next(ita)) 
# 创建迭代对象
class Container:
    def __init__(self, start = 0, end = 0):
        self.start = start
        self.end = end
    def __iter__(self):
        print("[LOG] I made this iterator!")
        return self
    def __next__(self):
        print("[LOG] Calling __next__ method!")
        if self.start < self.end:
            i = self.start
            self.start += 1
            return i
        else:
            raise StopIteration()
c = Container(0, 5)
for i in c:
    print(i)
