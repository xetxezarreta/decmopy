class CompRec:
    def __init__(self, id: int, value: float):
        self.id = id
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def compareTo(self, o: "CompRec"):
        if self.value > o.value:
            return 1
        if self.value < o.value:
            return -1
        return 0
