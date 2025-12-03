import sys

def main():
    solution = 0
    counter = 50

    for line in sys.stdin:
        line = line.strip()
        offset = int(line[1:])
        oldCounter = counter
        if line[0] == 'L':
            counter = (counter - offset)
        else:
            counter = (counter + offset)
        
        if counter > 0:
            solution += counter // 100
        else:
            solution += (-counter) // 100
            if oldCounter != 0:
                solution += 1
        counter = counter % 100
    print(solution)

if __name__ == "__main__":
    main()
