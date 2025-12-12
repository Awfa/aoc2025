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
        paths[n] = 0
    paths["out"] = 1
    frontier = []
    frontier.append("out")
    while len(frontier) > 0:
        node = frontier.pop()
        path = paths[node]
        if node in reverseGraph:
            for sourceNode in reverseGraph[node]:
                paths[sourceNode] += path
                outBounds[sourceNode] -= 1
                if outBounds[sourceNode] == 0:
                    frontier.append(sourceNode)
    print(paths["you"])


if __name__ == "__main__":
    main()
