import sys
from collections import deque

def findMinimumButtonsPressedForMachine(machine):
    indicatorTargets, diagrams, _ = machine
    lightAmount = len(indicatorTargets)
    # do a breadth-first search to find the minimum amount of buttons to press
    toSearch = deque()
    visited = set()

    for d in diagrams:
        toSearch.append((d, [False] * lightAmount, 0))
    while len(toSearch) > 0:
        diagramToSearch, currentState, presses = toSearch.popleft()
        if currentState == indicatorTargets:
            return presses

        for x in diagramToSearch:
            currentState[x] = not currentState[x]
        if tuple(currentState) in visited:
            continue
        else:
            visited.add(tuple(currentState))
        for d in diagrams:
            toSearch.append((d, list(currentState), presses + 1))
    return None
        
def main():
    machines = []

    # parse the machine descriptions from stdin
    for line in sys.stdin:
        if len(line) == 0:
            break
        # example line:
        # [.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
        # ^ indicator lights
        #        ^ wiring diagrams
        #                                         ^ joltage (unused)
        rawIndicatorTargets, rest = line.split("]", 1)
        rawWiringDiagrams, rawJoltage = rest.split("{", 1)
        indicatorTargets = []
        for c in rawIndicatorTargets[1:]:
            indicatorTargets.append(c == "#")
        diagrams = []
        for rawDiagram in rawWiringDiagrams.strip().split(" "):
            processed = rawDiagram.strip("()").split(",")
            diagram = map(int, processed)
            diagrams.append(list(diagram))
        joltages = list(map(int, rawJoltage[:-2].split(",")))

        machines.append((indicatorTargets, diagrams, joltages))
    sum = 0
    for m in machines:
        steps = findMinimumButtonsPressedForMachine(m)
        sum += steps
        print(steps)
    print(f"Total is {sum}")

if __name__ == '__main__':
    main()
