import sys
import time
import numpy

def machineToMatrixForm(machine):
    _, diagrams, joltages = machine

    matrix = numpy.zeros((len(joltages), len(diagrams)+1), dtype=int)
    for j in range(len(diagrams)):
        for d in diagrams[j]:
            matrix[d, j] = 1
    for j in range(len(joltages)):
        matrix[j, -1] = joltages[j]
    return matrix

def solveSimpleCoefficients(matrix, debug = False):
    # finds trivial solutions to coefficients
    # if there is a row of the form
    # 0 0 0 .. x .. 0 0 y
    # then we know x = y
    # returns None if a coefficient is forced to be < 0 or fractional
    # returns None if a row of coefficients has to equal non-zero
    # else, returns (sumOfSolvedCoefficients, matrix)
    #  - sum of solved coefficients is used to calculate how many steps / button presses it took to get to matrix
    #  - matrix is the left unsolved that needs more exploration
    sumOfSolvedCoefficients = None
    solvedCoefficients = [None] * (matrix.shape[1] - 1)
    solving = True
    while solving:
        solving = False
        for row in matrix:
            coefficients = row[:-1]
            nonZerosFound = 0
            nonZeroIdx = None

            # scan the coefficients in the row for a non zero
            for i, c in enumerate(coefficients):
                if c != 0:
                    if nonZeroIdx is None:
                        nonZeroIdx = i
                    nonZerosFound += 1
            if nonZerosFound > 1:
                continue
            if nonZeroIdx is None:
                if row[-1] != 0:
                    # a row of all coefficients of 0, with an equality to an non-zero answer means contradiction
                    return None
                continue
            remainder = row[-1] % row[nonZeroIdx]
            solution = int(row[-1] // row[nonZeroIdx])
            if remainder != 0 or solution < 0:
                return None
            if sumOfSolvedCoefficients is None:
                sumOfSolvedCoefficients = 0
            solvedCoefficients[nonZeroIdx] = solution
            sumOfSolvedCoefficients += solution
            pluggedInColumn = matrix[...,nonZeroIdx]*solution
            matrix[...,-1] -= pluggedInColumn
            matrix[...,nonZeroIdx] = 0
            solving = True

    return (sumOfSolvedCoefficients, solvedCoefficients, matrix)

# try to identify fixed variables
def optimizeMachine(matrix, debug = False):
    # lets try to find coefficients that are fixed - idea is to get this matrix to reduced row echelon form
    # try to get rows that have a single 1 within them

    # start doing eliminations
    ableToReduce = True
    totalSolvedSteps = 0
    coefficients = [None] * (matrix.shape[1] - 1)

    while ableToReduce:
        ableToReduce = False

        # for each coefficient slot (width - 1)
        targetRow = 0
        for coefficientIdx in range(matrix.shape[1] - 1):
            # find the first non zero within the column for the coefficient we're targetting
            firstNonZeroInColumn = None
            for row in range(targetRow, matrix.shape[0]):
                if matrix[row, coefficientIdx] != 0:
                    if firstNonZeroInColumn is None:
                        firstNonZeroInColumn = row
                        # break
                    elif abs(matrix[firstNonZeroInColumn, coefficientIdx]) > abs(matrix[row, coefficientIdx]):
                        firstNonZeroInColumn = row
            if firstNonZeroInColumn is None:
                continue

            # swap that found row so coefficient #x is on row x
            # if debug:
            #     print(f"Before swap of {targetRow} and {firstNonZeroInColumn}")
            #     prettyPrintMatrix(matrix, 3)
            matrix[[targetRow, firstNonZeroInColumn]] = matrix[[firstNonZeroInColumn, targetRow]]
            # if debug:
            #     print(f"After swap of {targetRow} and {firstNonZeroInColumn}")
            #     prettyPrintMatrix(matrix)
            #     print()

            # take the row and use it to simplify all the other rows
            for j in range(matrix.shape[0]):
                if j == targetRow:
                    continue
                if matrix[j, coefficientIdx] % matrix[targetRow, coefficientIdx] == 0:
                    matrix[j] -= matrix[targetRow] * (matrix[j, coefficientIdx] // matrix[targetRow, coefficientIdx])
                else:
                    pass
            # print(f"After using {targetRow} to simplify other rows")
            # prettyPrintMatrix(matrix)
            # print()
            targetRow += 1

        # scan for solvable coefficients
        x = solveSimpleCoefficients(matrix, debug)
        if x is not None:
            sumOfTotalSolvedSteps, cs, matrix = x
            if sumOfTotalSolvedSteps is not None:
                totalSolvedSteps += sumOfTotalSolvedSteps
                if sumOfTotalSolvedSteps > 0:
                    ableToReduce = True
            for i, c in enumerate(cs):
                if c is not None:
                    assert(coefficients[i] is None)
                    coefficients[i] = c
        else:
            return None
    return totalSolvedSteps, coefficients, matrix

def prettyPrintMatrix(matrix, tabs=0):
    for r in matrix:
        print(f"{"    " * tabs}{[int(c) for c in r]}")

def findMinimumSteps(matrix):
    # based off of my work in https://cse442-17f.github.io/Conflict-Driven-Clause-Learning/ !
    # so fun when I get to use my grad level class topics from over 7 years ago!! :)
    # For the given matrix, we have to find the minimum coefficients that satisfy the linear system
    # I initially started by doing a BFS / A* search exploring the possible coefficient values, but the search space is too large
    # One optimization we can take care of for this search, is similar to Boolean Contraint Propagation, but for numbers
    # So like, if A_1 * c_1 + A_2 * c_2 = B_1, if we fix c_1, we can instantly solve for c_2 using math
    # Additionally, we know we reached a deadend/conflict, if c_2 is negative as a result, or non integer, because amount of button presses are only positive integers
    # This is essentially DPLL but with positive integers!!
    #
    # tl;dr, I do a DFS to explore coefficient possibilities
    # Whenever I set a candidate coefficient, I use linear algebra to fix as many of the other coefficients as possible to reduce search space - or to find out I hit a contradiction
    # After the other coefficients are set, keep doing DFS
    # This explores the entire search space and finds the minimum solution

    # the original problem matrix is nice and all positive numbers, letting us bound our search candidates
    constraintMatrix = matrix.copy()

    def computeNewContraintMatrix(matrix, coefficientIdx, coefficient):
        matrix[:, -1] -= matrix[:, coefficientIdx] * coefficient
        matrix[:, coefficientIdx] = 0
        return matrix

    result = optimizeMachine(matrix)
    if result is None:
        print("bad coefficients: ", coefficients)
        return None
    
    steps, coefficients, matrix = result
    if None not in coefficients:
        return (0, steps)
    
    constraintMatrix = constraintMatrix.copy()
    for i, c in enumerate(coefficients):
        if c is not None:
            computeNewContraintMatrix(constraintMatrix, i, c)

    iterations = 0


    startTime = time.perf_counter_ns()
    lastTime = startTime
    isSlow = False

    globalSolution = None
    globalSolutionCoefficients = None

    def helper(matrix, coefficients: list[int], holes: list[int], constraintMatrix, currentSteps):
        # matrix is the optimized matrix made from the original matrix with the coefficients applied already
        # coefficients is our current list of coefficients
        # holes is a list of indicies in coefficients == None
        # constraintMatrix is our original matrix, with the plugged in coefficientMatrix
        nonlocal iterations
        nonlocal lastTime, isSlow
        nonlocal globalSolution, globalSolutionCoefficients
        
        if globalSolution is not None and currentSteps >= globalSolution:
            return None
        optimalAmountOfSteps = max(map(int, constraintMatrix[:, -1]))
        if globalSolution is not None and currentSteps + optimalAmountOfSteps >= globalSolution:
            # print("Early return")
            return None

        assert(len(holes) > 0)
        holeIdx = holes[0]
        holeMax = min(constraintMatrix[j, -1] for j, r in enumerate(constraintMatrix[:, holeIdx]) if r > 0)

        solutionFound = None
        for candidate in range(holeMax + 1):
            debug = False
            if globalSolution is not None and candidate + currentSteps >= globalSolution:
                break
            iterations += 1
            coefficients[holeIdx] = candidate
            candidateMatrix = matrix.copy()
            candidateMatrix[:, -1] -= candidateMatrix[:, holeIdx] * candidate
            candidateMatrix[:, holeIdx] = 0
            if debug:
                prettyPrintMatrix(candidateMatrix, 2)
                print()
            result = optimizeMachine(candidateMatrix, debug)
            if debug:
                print(result)
            if result is None:
                continue
            steps, coefficientsFoundInCandidate, optimizedCandidateMatrix = result

            candidateConstraintMatrix = constraintMatrix.copy()
            computeNewContraintMatrix(candidateConstraintMatrix, holeIdx, candidate)
            mergedCoefficients = list(coefficients)
            for i, c in enumerate(coefficientsFoundInCandidate):
                if c is not None:
                    assert(coefficients[i] is None)
                    mergedCoefficients[i] = c
                    computeNewContraintMatrix(candidateConstraintMatrix, i, c)
            totalStepsToOptimizedCandidateMatrix = currentSteps + steps + candidate
            # print("    Candidates led to solution", mergedCoefficients, "in", totalStepsToOptimizedCandidateMatrix, "steps")
            
            totalSteps = None
            mergedCoefficientHoles = [i for i, c in enumerate(mergedCoefficients) if c is None]
            if len(mergedCoefficientHoles) > 0:
                result = helper(optimizedCandidateMatrix, mergedCoefficients, mergedCoefficientHoles, candidateConstraintMatrix, totalStepsToOptimizedCandidateMatrix)
                if result is not None:
                    totalSteps = result
                else:
                    continue
            else:
                totalSteps = totalStepsToOptimizedCandidateMatrix
            if solutionFound is None or totalSteps < solutionFound:
                solutionFound = totalSteps
            if globalSolution is None or totalSteps < globalSolution:
                globalSolution = totalSteps
                globalSolutionCoefficients = mergedCoefficients

        coefficients[holeIdx] = None
        # if memoKey not in memo:
        #     memo[memoKey] = solutionFound
        currentTime = time.perf_counter_ns()
        if currentTime - lastTime > 30*1000000000:
            lastTime = currentTime
            print(f"  Current Global Solution = {globalSolution}")
            print(f"  On iteration {iterations} - just finished looking at coefficients {coefficients} ({currentSteps}) at hole index {holeIdx}..")
            print(f"    Found solution here? ", solutionFound)
            prettyPrintMatrix(constraintMatrix, 1)
            # print(optimizeMachine(constraintMatrix.copy()))
            print()
            prettyPrintMatrix(matrix, 1)
            # print(optimizeMachine(matrix.copy()))
            isSlow = True
            # print(holes)
            # print(constraintMatrix)
            # print(matrix)
        return globalSolution

    holes = [i for i, c in enumerate(coefficients) if c is None]
    result = helper(matrix, coefficients, holes, constraintMatrix, steps)

    now = time.perf_counter_ns()
    print(f"  Finished in {(now - startTime) / (1000000000 / 1000)}ms")
    return (iterations, result)

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
    machines = machines
    startTime = time.perf_counter_ns()
    for i, m in enumerate(machines):
        print(f"Working on machine {i}")
        iters, steps = findMinimumSteps(machineToMatrixForm(m))
        if iters == 0:
            print(f"Solved {steps} total steps needed")
        else:
            print(f"Solved {steps} total steps needed in {iters} iterations")
        sum += steps
    endTime = time.perf_counter_ns()
    elapsed = endTime - startTime
    elapsedInSecs = elapsed / 1000000000
    print(f"Solved in {elapsedInSecs} seconds!")
    print(f"Total is {sum}")

if __name__ == '__main__':
    main()
