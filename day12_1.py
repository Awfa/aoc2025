import sys
import numpy as np
import time
from tkinter import *
from tkinter import ttk
import threading
import queue
import random

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

    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            if shape[i, j] and subRegion[i, j]:
                return False
    subRegion += shape
    return True

def removeShapeFromRegion(region: np.typing.NDArray, shape: np.typing.NDArray, coordinate: tuple[int, int]):
    y, x = coordinate
    subRegion = region[y:y + shape.shape[0], x:x + shape.shape[1]]
    subRegion ^= shape

def computeBoundingBoxSize(region) -> np.typing.NDArray:
    """Return the bounding (height,width) of the region where the region is non-zero"""
    nonzeros = np.nonzero(region)
    mins = np.min(nonzeros, axis=1)
    maxs = np.max(nonzeros, axis=1)
    bounds = maxs - mins + np.array([1, 1])
    return bounds

def getShapeVolume(shape: np.typing.NDArray) -> int:
    return int(np.count_nonzero(shape))

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
            # this is incorrect, the top left origin of the shape can share this square
            # this works if the top left of the shape has a hole there
            # if region[y, x] == 0b10:
            #     continue

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

    bestBoundingBoxes = dict() # map from bounding box -> map coordinate -> (shapeIdx, rotationIdx)
    cache = dict()
    for shapeIdx, testShape in enumerate(shapeRotations):
        for rotationIdx, testShapeRotation in enumerate(testShape):
            cacheKey = (shapeIdx, rotationIdx)
            for testCoordinate in resultCoordinates:
                c = tuple(map(int, testCoordinate))
                if not addShapeToRegion(region, testShapeRotation, c):
                    continue
                boundingBox = computeBoundingBoxSize(region)
                removeShapeFromRegion(region, testShapeRotation, c)

                goodBounds = True
                boundingBoxesToDelete = []
                for bestBounds in bestBoundingBoxes.keys():
                    bbMatrix = np.array(list(bestBounds))
                    if boundingBox[0] < bbMatrix[0] and boundingBox[1] < bbMatrix[1] :
                        # ours is strictly better
                        boundingBoxesToDelete.append(bestBounds)
                    elif boundingBox[0] > bbMatrix[0] and boundingBox[1] > bbMatrix[1]:
                        # ours is worse than an existing
                        goodBounds = False
                        break
                for boundingBoxToDelete in boundingBoxesToDelete:
                    for coordinate in bestBoundingBoxes[boundingBoxToDelete].keys():
                        cacheKeyToDelete = bestBoundingBoxes[boundingBoxToDelete][coordinate]
                        coordinateAsCacheKey = (coordinate[0] - SHAPE_SIZE, coordinate[1] - SHAPE_SIZE)
                        cache[cacheKeyToDelete].remove(coordinateAsCacheKey)
                    del bestBoundingBoxes[boundingBoxToDelete]
                if goodBounds:
                    key = tuple(map(int, boundingBox))
                    if key not in bestBoundingBoxes:
                        bestBoundingBoxes[key] = dict()
                    bestBoundingBoxes[key][c] = cacheKey
                else:
                    continue

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

def prettyPrintIntRegion(region):
    for row in region:
        print("|", end="")
        for n in row:
            if n == 0:
                n = " "
            print(n, end="")
        print("|")
    print()


def visualizeHistoryStack(regionShape, historyStack, shapeRotations, fromIdx = None):
    """
    prints out the region made from the given history stack visualizing each new shape as an int
    this probably only works when there's less than 10 for now, formatting wise
    """
    visual = np.zeros(regionShape, dtype=int)
    fromIdx = len(historyStack) - 1
    for i, h in enumerate(historyStack):
        hCoordinate, (hShapeIdx, hRotationIdx) = h
        assert(addShapeToRegion(visual, shapeRotations[hShapeIdx][hRotationIdx]*(i+1), hCoordinate))
        if i >= fromIdx:
            prettyPrintIntRegion(visual)

def getRegionSymmetries(region: np.typing.NDArray):
    yield(tuple(map(int, np.ravel(region))))
    if region[0, 0] == region[-1, -1] or region[0, 0] == region[0, -1] or region[0, 0] == region[-1, 0]:
        one180Rotated = np.rot90(region, k = 2)
        yield(tuple(map(int, np.ravel(one180Rotated))))
        yield(tuple(map(int, np.ravel(np.fliplr(region)))))
        yield(tuple(map(int, np.ravel(np.flipud(region)))))
        if region.shape[0] == region.shape[1]:
            region = np.rot90(region, k = 1)
            yield(tuple(map(int, np.ravel(region))))
            one180Rotated = np.rot90(region, k = 2)
            yield(tuple(map(int, np.ravel(one180Rotated))))
            yield(tuple(map(int, np.ravel(np.fliplr(region)))))
            yield(tuple(map(int, np.ravel(np.flipud(region)))))

def check(shapeRotations: list[list[np.typing.NDArray]], regionConstraint: tuple[tuple[int, int], list[int]], neighborsCache: dict[tuple[int, int], dict[tuple[int, int], set[tuple[int, int]]]], shapeVolumes: list[int], outputQueue) -> bool:
    (regionWidth, regionHeight), presentCounts = regionConstraint

    emptyVolume = regionWidth * regionHeight
    requiredVolume = sum(shapeVolumes[i] * cnt for i, cnt in enumerate(presentCounts))
    if emptyVolume < requiredVolume:
        print(f"Empty volume {emptyVolume} < Required volume {requiredVolume}")
        return False

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

    startTime = time.perf_counter_ns()
    lastTime = startTime
    iterations = 0
    symmetriesIgnored = 0
    boundingBoxesIgnored = 0
    lowestBacktrackedSinceLastDraw = 0
    frontiers: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for shapeIdx in range(len(shapeRotations)):
        if presentCounts[shapeIdx] == 0:
            continue
        for rotationIdx, shape in enumerate(shapeRotations[shapeIdx]):
            frontiers.append(((0,0), (shapeIdx, rotationIdx), 0))

    visited = set()
    bestBoundingBoxes = dict() # map from tuple of shapes added to set of best bounding box sizes
    historyStack = []
    while len(frontiers) > 0:
        currentTime = time.perf_counter_ns()
        elapsed = currentTime - lastTime
        if elapsed > 1*1000000000:
            elapsedSeconds = (currentTime - startTime) / 1000000000
            outputQueue.put((region.shape, historyStack.copy(), lowestBacktrackedSinceLastDraw, iterations, elapsedSeconds, len(visited), len(frontiers), symmetriesIgnored, boundingBoxesIgnored))
            lastTime = currentTime
            lowestBacktrackedSinceLastDraw = len(historyStack)

        iterations += 1
        coordinate, (shapeIdx, rotationIdx), historyStackLen = frontiers.pop()
        shape = shapeRotations[shapeIdx][rotationIdx]
        
        backtracked = False
        while len(historyStack) > historyStackLen:
            # backtrack
            hCoordinate, (hShapeIdx, hRotationIdx) = historyStack.pop()
            removeShapeFromRegion(region, shapeRotations[hShapeIdx][hRotationIdx], hCoordinate)
            presentCounts[hShapeIdx] += 1
            backtracked = True
        if backtracked:
            lowestBacktrackedSinceLastDraw = min(lowestBacktrackedSinceLastDraw, historyStackLen)
        # if backtracked:
        #     visualizeHistoryStack(region.shape, historyStack, shapeRotations)
        #     visual = region * 4
        #     currentExplorationLevel = set()
        #     for f in frontiers:
        #         if f[2] == historyStackLen:
        #             currentExplorationLevel.add(f[0])
        #             if visual[f[0]] & 4:
        #                 visual[f[0]] = 6
        #             else:
        #                 visual[f[0]] = 1
        #     tester = np.full((SHAPE_SIZE, SHAPE_SIZE), True, dtype=bool)
        #     for s in shapeRotations:
        #         for r in s:
        #             tester &= r

        #     prettyPrintIntRegion(visual)
        #     prettyPrintIntRegion(tester * 1)
        #     return

        # see if a shape fits
        if not addShapeToRegion(region, shape, coordinate):
            continue

        historyStack.append((coordinate, (shapeIdx, rotationIdx)))
        presentCounts[shapeIdx] -= 1

        # see if we visited this or the symmetries before
        visitedBefore = False
        for i, vk in enumerate(getRegionSymmetries(region)):
            if vk in visited:
                if i > 0:
                    symmetriesIgnored += 1
                visitedBefore = True
                break
        visited.add(next(getRegionSymmetries(region)))
        if visitedBefore:
            continue

        historicalShapesKey = tuple(h[1][0] for h in historyStack)
        currentBoundingBox = computeBoundingBoxSize(region)
        if historicalShapesKey not in bestBoundingBoxes:
            bestBoundingBoxes[historicalShapesKey] = set()
        goodBounds = True
        boundingBoxesToDelete = []
        for bestBoundingBox in bestBoundingBoxes[historicalShapesKey]:
            bbMatrix = np.array(list(bestBoundingBox))
            if currentBoundingBox[0] < bbMatrix[0] and currentBoundingBox[1] < bbMatrix[1]:
                boundingBoxesToDelete.append(bestBoundingBox)
            elif currentBoundingBox[0] > bbMatrix[0] and currentBoundingBox[1] > bbMatrix[1]:
                # our current shape configuration bounding box is strictly worse
                goodBounds = False
                break
        if not goodBounds:
            boundingBoxesIgnored += 1
            continue
        bestBoundingBoxes[historicalShapesKey].add(tuple(map(int, currentBoundingBox)))
        for boundingBoxToDelete in boundingBoxesToDelete:
            bestBoundingBoxes[historicalShapesKey].remove(boundingBoxToDelete)


        # Scan for end condition - we fit all the shapes!
        nonZeroFound = False
        for c in presentCounts:
            if c != 0:
                nonZeroFound = True
                break
        if not nonZeroFound:
            visualizeHistoryStack(region.shape, historyStack, shapeRotations, len(historyStack)-1)
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
    return False

def generate_data(outputQueue, shapes, regionConstraints, shapeRotations):
    shapeVolumes = [getShapeVolume(s) for s in shapes]
    
    # dbg = findShapeFrontier(shapes[0], shapeRotations)
    # for k in dbg.keys():
    #     if k != (3,1):
    #         continue
    #     for c in dbg[k]:
    #         c = (c[0] + SHAPE_SIZE, c[1] + SHAPE_SIZE)
    #         historyStack = []
    #         historyStack.append(((SHAPE_SIZE, SHAPE_SIZE), (0, 0)))
    #         historyStack.append((c, k))
    #         visualizeHistoryStack((SHAPE_SIZE*3, SHAPE_SIZE*3), historyStack, shapeRotations)
    # print(sum(len(d) for d in dbg.values()))
    # return
    neighborsCache = computeShapeNeighborsCache(shapeRotations)
    print("Pairings between shapes", sum(len(innerV) for v in neighborsCache.values() for innerV in v.values()))
    for i, constraint in enumerate(regionConstraints[0:3]):
        result = check(shapeRotations, constraint, neighborsCache, shapeVolumes, outputQueue)
        print(f"Tree{i} = {result}")

CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 1024
class CanvasApp:
    def __init__(self, root, inputQueue, shapeRotations):
        self.root = root
        self.queue = inputQueue
        self.shapeRotations = shapeRotations
        
        self.root.title("Tkinter Canvas Example")

        self.lastRegionShape = None
        self.lastRegionTag = None
        self.lastHistoryStack = None

        self.leftText = None
        self.rightText = None

        def get_random_color():
            # Generates a random color in #RRGGBB format
            return f"#{random.randint(0, 0xFFFFFF):06x}"

        self.colors = []
        for _ in range(1000):
            self.colors.append(get_random_color())
        # Create the canvas widget
        self.canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
        self.canvas.pack(padx=20, pady=20)
        self.process_queue()
    def process_queue(self):
        queueItem = None
        lowestBacktrackedSinceLastDraw = None
        try:
            while True:
                queueItem = self.queue.get_nowait()
                if lowestBacktrackedSinceLastDraw is not None:
                    lowestBacktrackedSinceLastDraw = min(lowestBacktrackedSinceLastDraw, queueItem[2])
                else:
                    lowestBacktrackedSinceLastDraw = queueItem[2]
        except queue.Empty:
            pass
        if queueItem is None:
            self.root.after(100, self.process_queue)

            return
        regionShape, historyStack, _, iterations, elapsedSeconds, lenVisited, lenFrontiers, symmetriesIgnored, boundingBoxesIgnored = queueItem
        
        scaleFactor = min(CANVAS_WIDTH, CANVAS_HEIGHT) / max(regionShape) * 0.80
        x1 = (CANVAS_WIDTH) / 2 - regionShape[1] / 2
        y1 = (CANVAS_HEIGHT) / 2 - regionShape[0] / 2
        if self.lastRegionShape is None:
            self.regionTag = self.canvas.create_rectangle(x1, y1, x1 + regionShape[1], y1 + regionShape[0], outline="black")
            self.canvas.scale(self.regionTag, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, scaleFactor, scaleFactor)
        elif self.lastRegionShape != regionShape:
            self.canvas.delete(self.regionTag)
            self.regionTag = self.canvas.create_rectangle(x1, y1, x1 + regionShape[1], y1 + regionShape[0], outline="black")
            self.canvas.scale(self.regionTag, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, scaleFactor, scaleFactor)
            self.lastRegionShape = regionShape

        deleteStart = min(len(historyStack), lowestBacktrackedSinceLastDraw)
        if self.lastHistoryStack is not None:
            # if len(self.lastHistoryStack) - lowestBacktrackedSinceLastDraw > 0:
            for j in range(deleteStart, len(self.lastHistoryStack)):
                self.canvas.delete(f"shape{j}")
        #     print("last history", len(self.lastHistoryStack))
        # print("     history", len(historyStack))
        self.lastHistoryStack = historyStack

        for i in range(deleteStart, len(historyStack)):
            coordinate, (shapeIdx, shapeRotationIdx) = historyStack[i]
            shape = self.shapeRotations[shapeIdx][shapeRotationIdx]
            # print(f"Adding shape {i}")
            shapeStrKey = f"shape{i}"
            for y in range(shape.shape[0]):
                for x in range(shape.shape[1]):
                    if shape[y, x]:
                        originX = x1 + coordinate[1] + x
                        originY = y1 + coordinate[0] + y

                        self.canvas.create_rectangle(originX, originY, originX+1, originY+1, fill=self.colors[i], outline=self.colors[i], tags=shapeStrKey)
            self.canvas.scale(shapeStrKey, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, scaleFactor, scaleFactor)
    
        if self.leftText is None:
            self.leftText = self.canvas.create_text(x1, y1 + regionShape[0], 
                text=f"Iterations = {iterations}\nElapsed = {elapsedSeconds} seconds\nVisited = {lenVisited}",
                anchor="nw", tags="info")
            self.rightText = self.canvas.create_text(x1 + regionShape[1], y1 + regionShape[0],
                text=f"Frontiers = {lenFrontiers}\nSymmetries ignored = {symmetriesIgnored}\nBounding boxes ignored= {boundingBoxesIgnored}",
                anchor="ne", tags="info")
            self.canvas.scale("info", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, scaleFactor, scaleFactor)
        else:
            self.canvas.itemconfig(self.leftText, text=f"Iterations = {iterations}\nElapsed = {elapsedSeconds} seconds\nVisited = {lenVisited}")
            self.canvas.itemconfig(self.rightText, text=f"Frontiers = {lenFrontiers}\nSymmetries ignored = {symmetriesIgnored}\nBounding boxes ignored= {boundingBoxesIgnored}")

        self.root.after(100, self.process_queue)

def main():
    shapes, regionConstraints = parseInput(sys.stdin)
    shapeRotations = computeShapeRotations(shapes)

    outputQueue = queue.Queue(maxsize=10)
    workerThread = threading.Thread(daemon=True, target=generate_data, args=(outputQueue, shapes, regionConstraints, shapeRotations))
    workerThread.start()
    root = Tk()
    app = CanvasApp(root, outputQueue, shapeRotations)
    root.mainloop()

if __name__ == '__main__':
    main()
