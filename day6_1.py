import sys
import functools

def main():
    rows = []
    for line in sys.stdin:
        row = line.split()
        if len(row) == 0:
            break
        rows.append(row)

    solution = 0
    for column in range(len(rows[0])):
        opName = rows[-1][column]
        operator = None
        if opName == '+':
            operator = lambda a, b: a + b
        elif opName == '*':
            operator = lambda a, b: a * b
        else:
            print(f"UNKNOWN OPERATOR {opName}")
            return
        
        numbers = [int(a[column]) for a in rows[0:-1]]
        print(f"{numbers} {opName} = ", end = "")
        value = functools.reduce(operator, numbers[1:], numbers[0])
        print(value)

        solution += value
    print(solution)
            

if __name__ == '__main__':
    main()