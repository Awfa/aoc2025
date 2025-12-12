import sys
import numpy as np
import time

SHAPE_SIZE = 3

def parseInput(input) -> tuple[list[np.typing.NDArray], list[tuple[tuple[int, int], list[int]]]]:
    shapes = []
    regionConstraints = []
    parsingShapes = True
    inShape = False
    for line in input:
        line = line.strip()
        if inShape:
            if len(line) != 0:
                shapes[-1].append([c == '#' for c in line])
            else:
                shapes[-1] = np.array(shapes[-1], bool)
                inShape = False
            continue
        if parsingShapes:
            if "x" in line:
                parsingShapes = False
            else:
                inShape = True
                shapes.append([])
                continue
        if len(line) == 0:
            break
        regionDimension, shapeCounts = line.split(":")
        width, height = map(int, regionDimension.split("x"))
        shapeCounts = list(map(int, shapeCounts.strip().split(" ")))
        regionConstraints.append(((width, height), shapeCounts))

    return shapes, regionConstraints

def computeShapeRotations(shapes: list[np.typing.NDArray]) -> list[list[np.typing.NDArray]]:
    shapeRotations = []
    for shape in shapes:
        seen = set()
        rotations = []
        for i in range(4):
            rotated = np.rot90(shape, k=i)
            key = tuple(map(int, rotated.ravel()))
            if key not in seen:
                seen.add(key)
                rotations.append(rotated)
        for i in range(2):
            for rotation in rotations:
                flipped = np.flip(rotation, axis = i)
                key = tuple(map(int, flipped.ravel()))
                if key not in seen:
                    seen.add(key)
                    rotations.append(flipped)
        shapeRotations.append(rotations)
    return shapeRotations

def addShapeToRegion(region: np.typing.NDArray, shape: np.typing.NDArray, coordinate: tuple[int, int]) -> bool:
    """
    tries to update region with the shape
    if it can't, returns false with the region untouched
    if it can, returns true with the region updated
    """
    y, x = coordinate
    subRegion = region[y:y + shape.shape[0], x:x + shape.shape[1]]

    if not np.any(subRegion & shape):
        subRegion += shape
        return True
    return False

def removeShapeFromRegion(region: np.typing.NDArray, shape: np.typing.NDArray, coordinate: tuple[int, int]):
    y, x = coordinate
    subRegion = region[y:y + shape.shape[0], x:x + shape.shape[1]]
    subRegion ^= shape

def findShapeFrontier(shape: np.typing.NDArray, shapeRotations: list[list[np.typing.NDArray]]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """
    Returns a map from a particular shape rotation, to a set of coord offsets where that shape rotation could fit up against the given shape
    """
    assert(shape.shape == (SHAPE_SIZE, SHAPE_SIZE))
    emptySpaceOffsets = set()
    for y in range(shape.shape[0]):
        for x in range(shape.shape[1]):
            if shape[y, x]:
                up = (y - 1, x)
                right = (y, x + 1)
                down = (y + 1, x)
                left = (y, x - 1)
                if y == 0 or not shape[up]:
                    emptySpaceOffsets.add(up)
                if y == shape.shape[0] - 1 or not shape[down]:
                    emptySpaceOffsets.add(down)
                if x == 0 or not shape[left]:
                    emptySpaceOffsets.add(left)
                if x == shape.shape[1] - 1 or not shape[right]:
                    emptySpaceOffsets.add(right)
    
    # simulate placing a 3x3 'tester' block to cull candidate neighbor origins
    # we make a region where holes are represented as 0b01, the shape is 0b10, and our tester will be 0b10
    region = np.zeros((SHAPE_SIZE * 3, SHAPE_SIZE * 3), dtype=int)
    for offset in emptySpaceOffsets:
        y, x = offset
        y += SHAPE_SIZE
        x += SHAPE_SIZE
        region[y, x] = 0b01
    # place the shape in the middle
    addShapeToRegion(region, shape * 0b10, (SHAPE_SIZE, SHAPE_SIZE))

    result = np.zeros(region.shape, dtype=int)
    tester = np.full((SHAPE_SIZE, SHAPE_SIZE), 0b10)
    for y in range(region.shape[0] - SHAPE_SIZE + 1):
        for x in range(region.shape[1] - SHAPE_SIZE + 1):
            if region[y, x] == 0b10:
                continue

            testResult = tester - region[y:y+SHAPE_SIZE, x:x+SHAPE_SIZE]
            # the tester should intersect with a hole, in that case, the matrix will have a 1 since 2 - 1 = 1
            if not (testResult == 1).any():
                continue
            # parts where the tester is 0 is where it's intersecting with 'shape'
            # this is okay, but there can't be too much intersection that it is unfeasible for another shape to be placed
            # a heuristic for this can be if there is a 2 in every column and a 2 in every row
            # this heuristic targets the fact that every input shape fills the 3x3 bounding box
            # a 2 in every column means the tester is 3 wide - if there isn't, then it is < 3 wide which can't fit a 3x3 shape
            # a 2 in every row means the tester is 3 tall - if there isn't, then it is < 3 tall which can't fit a 3x3 shape
            twos = testResult > 0
            if not twos.all(0).any() or not twos.all(1).any():
                continue
            result[y, x] = 1

    region = np.zeros(region.shape, dtype=bool)
    addShapeToRegion(region, shape, (SHAPE_SIZE, SHAPE_SIZE))
    resultCoordinates = np.transpose(np.nonzero(result))

    cache = dict()
    for shapeIdx, testShape in enumerate(shapeRotations):
        for rotationIdx, testShapeRotation in enumerate(testShape):
            cacheKey = (shapeIdx, rotationIdx)
            for testCoordinate in resultCoordinates:
                c = tuple(map(int, testCoordinate))
                if addShapeToRegion(region, testShapeRotation, c):
                    # There is still room for improvement here:
                    # while bounding boxes touch the holes, there can still be a gap after the shape is placed
                    # Example:
                    # 222---
                    # 22-111
                    # 22-11-
                    # ---11-
                    # Here the '2' shape is placed and covers some holes left of '1', but there's a gap still
                    # This is because our 'test' array wasn't a shape, but just a 3x3 brick
                    removeShapeFromRegion(region, testShapeRotation, c)
                    if cacheKey not in cache:
                        cache[cacheKey] = set()
                    # translate back to the shape's origin coordinates
                    # we placed the shape at (SHAPE_SIZE, SHAPE_SIZE), so we subtract that
                    cache[cacheKey].add((c[0] - SHAPE_SIZE, c[1] - SHAPE_SIZE))
    return cache

def computeShapeNeighborsCache(shapeRotations: list[list[np.typing.NDArray]]) -> dict[tuple[int, int], dict[tuple[int, int], set[tuple[int, int]]]]:
    neighborOffsetCache = dict()
    for shapeIdx, shapes in enumerate(shapeRotations):
        for rotationIdx, rotations in enumerate(shapes):
            neighborOffsetCache[(shapeIdx, rotationIdx)] = findShapeFrontier(rotations, shapeRotations)
    return neighborOffsetCache

def check(shapeRotations: list[list[np.typing.NDArray]], regionConstraint: tuple[tuple[int, int], list[int]], neighborsCache: dict[tuple[int, int], dict[tuple[int, int], set[tuple[int, int]]]]) -> bool:
    (regionWidth, regionHeight), presentCounts = regionConstraint
    region = np.zeros((regionHeight, regionWidth), dtype=bool)
    # region under christmas tree
    #   <--- width --->
    # ^ (0, 0)
    # | height
    # |
    # .
    #
    # coordinate space is (units from top, units from left) (origin is top left)

    # frontiers are available spots we can try putting the top left of a shape in
    # DFS to explore if there's a possibility
    # print(region[0:3, 0:3])

    def visualizeHistoryStack(regionShape, historyStack, fromIdx = None):
        """
        prints out the region made from the given history stack visualizing each new shape as an int
        this probably only works when there's less than 10 for now, formatting wise
        """
        visual = np.zeros(regionShape, dtype=int)
        fromIdx = len(historyStack) - 1
        for i, h in enumerate(historyStack):
            hCoordinate, (hShapeIdx, hRotationIdx) = h
            addShapeToRegion(visual, shapeRotations[hShapeIdx][hRotationIdx]*(i+1), hCoordinate)
            if i >= fromIdx:
                for row in visual:
                    for n in row:
                        print(n, end="")
                    print()
                print()

    startTime = time.perf_counter_ns()
    lastTime = startTime
    iterations = 0
    frontiers: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for shapeIdx in range(len(shapeRotations)):
        if presentCounts[shapeIdx] == 0:
            continue
        for rotationIdx, shape in enumerate(shapeRotations[shapeIdx]):
            frontiers.append(((0,0), (shapeIdx, rotationIdx), 0))

    visited = set()
    historyStack = []
    while len(frontiers) > 0:
        iterations += 1
        coordinate, (shapeIdx, rotationIdx), historyStackLen = frontiers.pop()
        shape = shapeRotations[shapeIdx][rotationIdx]
        
        while len(historyStack) > historyStackLen:
            # backtrack
            hCoordinate, (hShapeIdx, hRotationIdx) = historyStack.pop()
            removeShapeFromRegion(region, shapeRotations[hShapeIdx][hRotationIdx], hCoordinate)
            presentCounts[hShapeIdx] += 1
        # see if a shape fits
        r = addShapeToRegion(region, shape, coordinate)
        if r:
            historyStack.append((coordinate, (shapeIdx, rotationIdx)))
            presentCounts[shapeIdx] -= 1

            # see if we visited this or the symmetries before
            visitedKeys = []
            for i in range(2):
                rotated = np.rot90(region, k = i*2)
                flip1 = np.fliplr(rotated)
                flip2 = np.flipud(rotated)
                visitedKeys.append(tuple(map(int, np.ravel(rotated))))
                visitedKeys.append(tuple(map(int, np.ravel(flip1))))
                visitedKeys.append(tuple(map(int, np.ravel(flip2))))
            visitedBefore = False
            for vk in visitedKeys:
                if vk in visited:
                    visitedBefore = True
                    break
            visited.update(visitedKeys)
            if visitedBefore:
                continue
            

            # Scan for end condition - we fit all the shapes!
            nonZeroFound = False
            for c in presentCounts:
                if c != 0:
                    nonZeroFound = True
                    break
            if not nonZeroFound:
                visualizeHistoryStack(region.shape, historyStack, len(historyStack)-1)
                return True

            possibleNeighbors = dict()
            cache = neighborsCache[shapeIdx, rotationIdx]
            for key in cache:
                if presentCounts[key[0]] == 0:
                    continue
                for c in cache[key]:
                    c = (c[0] + coordinate[0], c[1] + coordinate[1])
                    if 0 <= c[0] <= (region.shape[0] - SHAPE_SIZE) and 0 <= c[1] <= (region.shape[1] - SHAPE_SIZE):
                        if key not in possibleNeighbors:
                            possibleNeighbors[key] = list()
                        possibleNeighbors[key].append(c)
            for possibleNeighbor in possibleNeighbors.keys():
                for possibleCoordinate in possibleNeighbors[possibleNeighbor]:
                    frontiers.append((possibleCoordinate, possibleNeighbor, historyStackLen + 1))
        else:
            currentTime = time.perf_counter_ns()
            if currentTime - lastTime > 5*1000000000:
                lastTime = currentTime
                elapsedSeconds = (currentTime - startTime) / 1000000000
                print(f"On iteration {iterations} after {elapsedSeconds} secs | Visited set size = {len(visited)} | frontier size = {len(frontiers)}")
                visualizeHistoryStack(region.shape, historyStack, len(historyStack)-1)

    return False


def main():
    shapes, regionConstraints = parseInput(sys.stdin)
    shapeRotations = computeShapeRotations(shapes)
    neighborsCache = computeShapeNeighborsCache(shapeRotations)

    for i, constraint in enumerate(regionConstraints):
        result = check(shapeRotations, constraint, neighborsCache)
        print(f"Tree{i} = {result}")

if __name__ == '__main__':
    main()
