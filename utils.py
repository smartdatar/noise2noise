
class Avg:
    def __int__(self):
        self.n = 0
        self.sum = 0


    def __call__(self, num):
        self.n += 1
        self.sum += num
        return self.sum / self.n
