import sys
import functools

# Cephalopod math is written right-to-left in columns. Each number is given in its own column, with the most significant digit at the top and the least significant digit at the bottom.
def transformProblem(problem):
    newProblem = [problem[0]]
    rawNumbers = problem[1:]
    for operand in reversed(range(len(rawNumbers[0]))):
        # each column is an operand
        x = 0
        for digit in range(len(rawNumbers)):
            # digits are in each row
            d = rawNumbers[digit][operand]
            if d != ' ':
                x *= 10
                x += int(d)
            elif x > 0:
                break
        newProblem.append(x)
    return newProblem

def main():
    rawLines = []
    for line in sys.stdin:
        if len(line) == 0:
            break
        rawLines.append(line)

    # parse operator line
    # operator is always on the left-most side, followed by spaces
    problems = []

    currentLength = None
    for (index, ch) in enumerate(rawLines[-1]):
        if ch != ' ':
            if currentLength != None:
                if ch == '\n':
                    index += 1
                    currentLength += 1
                for i in range(len(rawLines)-1):
                    problems[-1].append(rawLines[i][index - currentLength:index - 1])

            if ch != '\n':
                opName = ch
                if opName == '+':
                    operator = lambda a, b: a + b
                elif opName == '*':
                    operator = lambda a, b: a * b
                else:
                    print(f"UNKNOWN OPERATOR {opName}")
                    return
                problems.append([operator])
                currentLength = 1
        else:
            currentLength += 1
    problems = map(transformProblem, problems)
    calculated = map(lambda problem: functools.reduce(problem[0], problem[2:], problem[1]), problems)
    solution = sum(calculated)
    print(solution)            

if __name__ == '__main__':
    main()