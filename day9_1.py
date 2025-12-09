import sys

def area(a, b):
    return (abs(a[0] - b[0]) + 1) * (abs(a[1] - b[1]) + 1)

def main():
    redTiles = []
    for lines in sys.stdin:
        if len(lines) == 0:
            break
        x, y = map(int, lines.split(","))
        redTiles.append((x, y))
    areas = []
    for i in range(1, len(redTiles)):
        for j in range(len(redTiles)):
            areas.append((area(redTiles[i], redTiles[j]), (i, j)))

    maxArea, (t1, t2) = max(areas, key = lambda a: a[0])

    print(maxArea, redTiles[t1], redTiles[t2])

if __name__ == '__main__':
    main()
