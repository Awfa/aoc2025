import sys
import math
import itertools
import functools
import heapq

def distance(junction1, junction2):
    sum = 0
    for i in range(len(junction1)):
        sum += (junction1[i] - junction2[i]) ** 2
    return math.sqrt(sum)

def main():
    # parse into junctions
    junctions = []
    for line in sys.stdin:
        if len(line) == 0:
            break
        x, y, z = line.split(",")
        junctions.append((int(x), int(y), int(z)))

    # build distance min-heap
    distances = []
    for i in range(len(junctions) - 1):
        # print(" " * i * 9, end = "")
        for j in range(i + 1, len(junctions)):
            d = distance(junctions[i], junctions[j])
            distances.append((d, [i, j]))
            # print(f"{d: 8.2f} ", end = "")
        # print()
    heapq.heapify(distances)

    print("Pairs: ", len(distances))

    # connect 1000 pairs which are closest together, but not already connected
    connected = UnionFind(len(junctions))
    for _ in range(1000):
        d, closestPair = heapq.heappop(distances)
        i, j = closestPair
        if connected.union(i, j) is not None:
            print(f"New connection between {i} and {j}", junctions[i], junctions[j], f"distance = {d}")
        else:
            print(f"Already connected {i} and {j}", junctions[i], junctions[j], f"distance = {d}")
    sizes = dict()
    for i in range(len(connected)):
        id, size = connected.getSet(i)
        if id not in sizes:
            sizes[id] = size
    
    print(functools.reduce(lambda a, b: a * b, itertools.islice(sorted(sizes.values(), reverse = True), 3), 1))

class UnionFind:
    def __init__(self, numElements: int):
        self.graph = [[i, 1] for i in range(numElements)]
    
    def __len__(self):
        return len(self.graph)

    def getSet(self, elementIdx: int):
        ptr = self.graph[elementIdx][0]
        while ptr != self.graph[ptr][0]:
            newPtr = self.graph[self.graph[ptr][0]][0]
            self.graph[ptr][0] = newPtr
            ptr = newPtr
        self.graph[elementIdx][0] = ptr
        return [ptr, self.graph[ptr][1]]

    def union(self, elementIdx1: int, elementIdx2: int) -> int | None:
        root1, root1Size = self.getSet(elementIdx1)
        root2, root2Size = self.getSet(elementIdx2)
        if root1 != root2:
            _, root2Size = self.graph[root2]
            newSize = root1Size + root2Size
            self.graph[root2][0] = root1
            self.graph[root1][1] = newSize
            return newSize
        return None

if __name__ == '__main__':
    main()