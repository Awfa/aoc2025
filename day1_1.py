import sys

def main():
    solution = 0
    counter = 50

    for line in sys.stdin:
        line = line.strip()
        offset = int(line[1:])
        if line[0] == 'L':
            counter = (counter - offset) % 100
        else:
            counter = (counter + offset) % 100
        if counter == 0:
            solution += 1
    print(solution)

if __name__ == "__main__":
    main()