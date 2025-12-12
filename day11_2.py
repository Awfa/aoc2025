import sys

def main():
    graph = dict()
    for line in sys.stdin:
        if len(line) == 0:
            break
        node, edges = line.split(":")
        edges = set(edges.split())
        graph[node] = edges

    # reverse topological sort to count up paths
    paths = dict()
    outBounds = dict()
    reverseGraph = dict()
    for n in graph.keys():
        outBounds[n] = len(graph[n])
        for destinationNode in graph[n]:
            if destinationNode not in reverseGraph:
                reverseGraph[destinationNode] = set()
            reverseGraph[destinationNode].add(n)
        paths[n] = (0, 0, 0, 0)
    paths["out"] = (0, 0, 0, 1)
    frontier = []
    frontier.append("out")
    while len(frontier) > 0:
        node = frontier.pop()
        thruBothDest, thruDacDest, thruFftDest, thruNoneDest = paths[node]
        if node in reverseGraph:
            for sourceNode in reverseGraph[node]:
                thruBothSource, thruDacSource, thruFftSource, thruNoneSource = paths[sourceNode]
                if sourceNode == "dac":
                    thruBothSource += thruFftDest + thruBothDest
                    thruDacSource += thruNoneDest + thruDacDest
                elif sourceNode == "fft":
                    thruBothSource += thruDacDest + thruBothDest
                    thruFftSource += thruNoneDest + thruFftDest
                else:
                    thruBothSource += thruBothDest
                    thruDacSource += thruDacDest
                    thruFftSource += thruFftDest
                    thruNoneSource += thruNoneDest
                paths[sourceNode] = (thruBothSource, thruDacSource, thruFftSource, thruNoneSource)
                outBounds[sourceNode] -= 1
                if outBounds[sourceNode] == 0:
                    frontier.append(sourceNode)
    thruBothDest, thruDacDest, thruFftDest, thruNoneDest = paths["svr"]
    print(f"Through both dac and fft: {thruBothDest}")

if __name__ == "__main__":
    main()
