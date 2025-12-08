import sys
import itertools

def main():
    manifold = []
    for line in sys.stdin:
        if len(line) == 0:
            break
        manifold.append(line.strip())
    
    beams = set()
    beams.add(manifold[0].index('S'))
    split = 0
    for row in itertools.islice(manifold, 1, len(manifold)):
        newBeams = set()
        for beam in beams:
            if row[beam] == '^':
                newBeams.add(beam - 1)
                newBeams.add(beam + 1)
                split += 1
            else:
                newBeams.add(beam)
        beams = newBeams
    print(split)

if __name__ == '__main__':
    main()
