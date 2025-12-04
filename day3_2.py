import sys
import itertools

def bubbleDownDigits(candidate, digits):
    for i in range(len(digits)):
        if candidate >= digits[i]:
            digits[i], candidate = candidate, digits[i]
        else:
            break

def main():
    # reverse loop solution
    # same thing but generalized to twelve digits
    solution = 0
    for line in sys.stdin:
        line = line.strip()
        joltageDigits = list(map(int, line[-12:]))
        for i in map(int, itertools.islice(reversed(line), 12, None, None)):
            bubbleDownDigits(i, joltageDigits)
        number = 0
        for digit in joltageDigits:
            number = number * 10 + digit
        solution += number
        print(line, number)
    print(solution)

if __name__ == "__main__":
    main()
