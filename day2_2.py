import sys

def main():
    solutionSet = set()
    for line in sys.stdin:
        ranges = line.split(',')
        for r in ranges:
            ## Parsing
            start_str, end_str = r.split('-')
            start_str, end_str = start_str.strip(), end_str.strip()
            start_int, end_int = int(start_str), int(end_str)

            digits_in_start = len(start_str)
            digits_in_end = len(end_str)

            print(f"Range: {start_int} to {end_int}")
            for num_digits_to_repeat in range(1, digits_in_end // 2 + 1):
                repeat_times = []
                for x in range(digits_in_start, digits_in_end + 1):
                    if x % num_digits_to_repeat == 0 and x != num_digits_to_repeat:
                        repeat_times.append(x // num_digits_to_repeat)
                # print(f"  {repeat_times}")
                for repeat_time in repeat_times:
                    low = 1
                    hi = 9

                    for _ in range(1, num_digits_to_repeat):
                        low *= 10
                        hi = hi * 10 + 9
                    candidates = []
                    for candidatePrefix in range(low, hi +1):
                        candidate = candidatePrefix
                        for i in range(1, repeat_time):
                            candidate *= 10**num_digits_to_repeat
                            candidate += candidatePrefix
                        if candidate < start_int or candidate > end_int:
                            continue
                        candidates.append(candidate)
                    # print(f"{low}->{hi}", end = "")
                    # print(f" X {repeat_time} ")

                    # if len(candidates) > 0:
                    #     first = candidates[0]
                    #     last = candidates[-1]
                    #     print(f"{first} -> {last}")
                    solutionSet.update(candidates)

                # Range: 123 to 123456789
                # Repeating 1 digits. (3) (9): 3 - 9 times
                # x x x
                # x x x x
                # x x x x x
                # x x x x x x
                # x x x x x x x
                # x x x x x x x x
                # x x x x x x x x x
                # Repeating 2 digits. (3) (9): 2 - 4 times
                # xx xx
                # xx xx xx
                # xx xx xx xx
                # Repeating 3 digits. (3) (9): 1 - 3 times
                # xxx <- filter later
                # xxx xxx
                # xxx xxx xxx
                # Repeating 4 digits. (3) (9): 1 - 2 times
                # xxxx <- filter later
                # xxxx xxxx


    sum = 0
    for x in solutionSet:
        sum += x
    print(sum)

if __name__ == "__main__":
    main()
