import sys
import itertools

# For each beam, keep track how many paths could've reached it
# .......S.......
# .......1....... 1
# ......1^1...... 2 L, R
# ......1.1......
# .....1^2^1..... 4 LL, LR, RL, RR
# .....1.2.1.....
# ....1^3^3^1.... 8 LLL, LLR, LRL, LRR, RLL, RLR, RRL, RRR
# ...............

def main():
    manifold = []
    for line in sys.stdin:
        if len(line) == 0:
            break
        manifold.append(line.strip())
    
    beams = dict()
    beams[manifold[0].index('S')] = 1 # tuple represents (beam index, # of paths)

    for row in itertools.islice(manifold, 1, len(manifold)):
        newBeams = dict()
        for beamIdx in beams.keys():
            beamPaths = beams[beamIdx]
            def updateBeam(idx, paths):
                if idx not in newBeams:
                    newBeams[idx] = 0
                newBeams[idx] += paths
            if row[beamIdx] == '^':
                updateBeam(beamIdx - 1, beamPaths)
                updateBeam(beamIdx + 1, beamPaths)
            else:
                updateBeam(beamIdx, beamPaths)
        beams = newBeams
    totalPaths = 0
    for beamIdx in beams.keys():
        totalPaths += beams[beamIdx]
    print(totalPaths)

if __name__ == '__main__':
    main()
