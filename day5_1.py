import sys
import itertools
from bisect import bisect_right

def main():
    ranges = []
    availableIngredients = []
    for line in sys.stdin:
        line = line.strip()
        if len(line) == 0:
            break
        start, end = line.split("-")
        ranges.append([int(start), int(end)])
    ranges.sort(key = lambda a: a[0])

    for line in sys.stdin:
        line = line.strip()
        if len(line) == 0:
            break
        availableIngredients.append(int(line))

    # interval merging
    # ranges = [[1,5],[10,15]]
    # availableIngredients = [5, 6, 10]
    mergedRange = [ranges[0]]
    for start, end in itertools.islice(ranges, 1, len(ranges)):
        prior = mergedRange[-1]
        if start > prior[1]:
            mergedRange.append([start, end])
            continue
        mergedRange[-1][1] = max(mergedRange[-1][1], end)

    solution = 0
    for availableIngredient in availableIngredients:
        i = bisect_right(mergedRange, availableIngredient, key = lambda a: a[0])
        if i == 0:
            continue
        start, end = mergedRange[i-1]
        if start <= availableIngredient <= end:
            solution += 1
    print(solution)

if __name__ == "__main__":
    main()
