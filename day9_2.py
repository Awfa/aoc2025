import sys
import numpy
import bisect

def area(a, b):
    return (abs(a[0] - b[0]) + 1) * (abs(a[1] - b[1]) + 1)

def yieldBetweens(a, b):
    if a[0] != b[0]:
        if a[0] > b[0]:
            offset = (-1, 0)
        else:
            offset = (1, 0)
    elif a[1] != b[1]:
        if a[1] > b[1]:
            offset = (0, -1)
        else:
            offset = (0, 1)
    next = (a[0] + offset[0], a[1] + offset[1])
    while next != b:
        yield next
        next = (next[0] + offset[0], next[1] + offset[1])

def yieldNeighbors(x, max):
    above = (x[0]-1, x[1])
    if above[0] >= 0:
        yield above
    below = (x[0]+1, x[1])
    if above[0] < max[0]:
        yield below
    right = (x[0], x[1]+1)
    if right[1] < max[1]:
        yield right
    left = (x[0], x[1]-1)
    if left[1] >= 0:
        yield left

def main():
    redTiles = []
    for lines in sys.stdin:
        if len(lines) == 0:
            break
        x, y = map(int, lines.split(","))
        redTiles.append((y, x))

    # 2d array to keep track of tile color
    # transform the domain of the red tiles to minimize the 2d array size
    # compress the coordinate space
    uniqueXs = sorted(set(t[0] for t in redTiles))
    uniqueYs = sorted(set(t[1] for t in redTiles))

    def toGridCoords(tile):
        x = bisect.bisect_left(uniqueXs, tile[0])
        y = bisect.bisect_left(uniqueYs, tile[1])
        return (x, y)

    gridShape = (len(uniqueXs), len(uniqueYs))
    print(f"Constructing grid {gridShape}...")
    grid = numpy.zeros(gridShape, dtype=bool)

    print("Drawing shape...")
    # Add each tile and the inbetweens to the 2d grid
    first = toGridCoords(redTiles[0])
    grid[first] = True
    last = first
    for i in range(1, len(redTiles)):
        next = toGridCoords(redTiles[i])
        for c in yieldBetweens(last, next):
            grid[c] = True
        grid[next] = True
        last = next
    next = first
    for c in yieldBetweens(last, next):
        grid[c] = True

    # We will do a flood fill to mark the inside region
    # To do a flood fill, we need a seed point on the inside of the tile line.
    # To find a seed, I will find one of the points on the domain edge, and look 'inwards' from there
    # Since we're at the outside of the domain, the only direction, inward, should be inside the tile line
    print("Flood fill: finding seed...")
    seed = None
    for i in range(len(grid)):
        if grid[i][0] and not grid[i][1]:
            seed = (i, 1)
            break
        if grid[i][-1] and not grid[i][-2]:
            seed = (i, len(grid[i]) - 1 - 1)
            break
    if seed is None:
        for i in range(len(grid[0])):
            if grid[0][i] and not grid[1][i]:
                seed = (1, i)
                break
            if grid[-1][i] and not grid[-2][i]:
                seed = (len(grid[0]) - 1 - 1, i)
                break
    if seed is None:
        print("Couldn't find a seed to flood fill from")
        return

    def floodFill(grid, seed):
        toVisit = list()
        toVisit.append(seed)
        while len(toVisit) > 0:
            next = toVisit.pop()
            grid[next] = True
            for neighbor in yieldNeighbors(next, gridShape):
                if not grid[neighbor]:
                    toVisit.append(neighbor)
    print("Flood fill: flooding...")
    floodFill(grid, seed)

    print("Getting all areas...")
    areas = []
    for i in range(1, len(redTiles)):
        for j in range(len(redTiles)):
            areas.append((area(redTiles[i], redTiles[j]), (i, j)))
    areas = sorted(areas, reverse = True)

    print("Culling areas to find the biggest one...")
    for a, (i, j) in areas:
        c1, c2 = toGridCoords(redTiles[i]), toGridCoords(redTiles[j])
        sortedX = [*sorted([c1[0], c2[0]])]
        sortedY = [*sorted([c1[1], c2[1]])]

        fullBreak = False
        for x in range(sortedX[0], sortedX[1] + 1):
            for y in range(sortedY[0], sortedY[1] + 1):
                if not grid[x][y]:
                    fullBreak = True
            if fullBreak:
                break
        if not fullBreak:
            print(f"Biggest area is {a}")
            return

if __name__ == '__main__':
    main()
