import sys
import heapq
import multiprocessing as mp

# this is too slow for the day 10 input :(
def findMinimumButtonsPressedForMachineWithAStar(machine):
    _, diagrams, joltages = machine
    lightAmount = len(joltages)
    diagrams = sorted(diagrams, key = lambda d: len(d))
    # A* algorithm
    def distanceHeuristic(current):
        # this heurisitic is admissible: never overestimating the true cost
        # this is admissible because each button can only advance any of the joltages up by one
        # so we need to at least press x buttons where x is the biggest gap between our current - and the target joltages
        #
        # being admissible means the A* algorithm returns the optimal solution (not necessarily in the best time)
        return max((joltages[i] - current[i] for i in range(len(current))))

    searchSteps = 0
    currentState = [0] * lightAmount
    currentStateTuple = tuple(currentState)
    openSet = []
    openSet.append(((0, 0), currentStateTuple))
    openSetPresences = set()
    openSetPresences.add(currentStateTuple)
    currentlyKnownLowestStepsFromStart = dict()
    currentlyKnownLowestStepsFromStart[currentStateTuple] = 0
    bestGuessOfPathCost = dict()
    bestGuessOfPathCost[currentStateTuple] = distanceHeuristic(currentStateTuple)

    while len(openSet) > 0:
        searchSteps += 1
        _, currentStateTuple = heapq.heappop(openSet)
        if currentStateTuple in openSetPresences:
            openSetPresences.remove(currentStateTuple)
        else:
            # skip dups
            continue
        currentState = list(currentStateTuple)
        presses = currentlyKnownLowestStepsFromStart[currentStateTuple]
        if currentState == joltages:
            return (searchSteps, presses)

        for d in diagrams:
            invalid = False
            for x in d:
                currentState[x] += 1
                if currentState[x] > joltages[x]:
                    invalid = True
                    break
            if not invalid:
                # currentState is now a neighbor
                currentStateTuple = tuple(currentState)
                newPresses = presses + 1
                if currentStateTuple not in currentlyKnownLowestStepsFromStart or newPresses < currentlyKnownLowestStepsFromStart[currentStateTuple]:
                    currentlyKnownLowestStepsFromStart[currentStateTuple] = newPresses
                    bestGuessOfPathCost[currentStateTuple] = newPresses + distanceHeuristic(currentStateTuple)
                    heapq.heappush(openSet, ((bestGuessOfPathCost[currentStateTuple], -newPresses), currentStateTuple))
                    # -newPresses is to try to explore the furthest along state first
                    openSetPresences.add(currentStateTuple)
            # restore currentState to enumerate next neighbor
            for x in d:
                currentState[x] -= 1
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
    with mp.Pool(16) as p:
        for iterations, steps in p.imap(findMinimumButtonsPressedForMachineWithAStar, machines):
            sum += steps
            print(f"Used {iterations} steps to find minimum steps = {steps}")
    print(f"Total is {sum}")

if __name__ == '__main__':
    main()
