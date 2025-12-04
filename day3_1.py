import sys
import itertools

def main():
    # reverse loop solution
    # start with a solution made up of the last 2 digits
    # consider the last digit in the unknown region, i
    # [unknown digits]ixy
    # invariants to keep:
    # - x is the biggest possible leftmost digit in the considered numbers
    # - y is the biggest possible digit between where x is, and where the end is
    # if i is bigger than x, make x'=i, and y=max(y, x).
    # if i is less than x, don't change anything and proceed to look at next digit
    # [unknown digits]i[ignored digits 1]x[ignored digits 2]y[ignored digits 3]
    # if the first case is hit in the above, y=max(y, x) is sufficient to maintain the second invariant, because
    #   if it wasn't sufficient:
    #    - if there was a bigger number in [ignored digits 2], invariant 2 was not held up
    #    - if there was a bigger number in [ignored digits 3], invariant 2 was not held up
    #    - if there was a bigger number in [ignored digits 1], invariant 1 was not held up
    solution = 0
    for line in sys.stdin:
        line = line.strip()
        x = int(line[-2])
        y = int(line[-1])
        for i in map(int, itertools.islice(reversed(line), 2, None, None)):
            if i >= x:
                x, y = i, max(x, y)
        number = x*10 + y
        solution += number
        print(line, number)
    print(solution)

if __name__ == "__main__":
    main()
