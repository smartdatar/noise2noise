
class Avg:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.mean = 0

    def __call__(self, num):
        self.n += 1
        self.sum += num
        self.mean = self.sum / self.n
        return self.mean


