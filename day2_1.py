import sys

def main():
    solution = 0
    for line in sys.stdin:
        ranges = line.split(',')
        for r in ranges:
            start_str, end_str = r.split('-')
            start_str, end_str = start_str.strip(), end_str.strip()
            start_int, end_int = int(start_str), int(end_str)

            digits_in_start = len(start_str)
            digits_in_end = len(end_str)

            adjusted_start_int, adjusted_end_int = start_int, end_int
            if digits_in_start % 2 != 0:
                adjusted_start_int = 10 ** (digits_in_start) # 105 -> 10 ^ 3 = 1000
                digits_in_start += 1
            if digits_in_end % 2 != 0:
                adjusted_end_int = 10 ** (digits_in_end - 1) - 1 # 105 -> 10 ^ 2 - 1 = 99
                digits_in_end -= 1
            if adjusted_start_int >= adjusted_end_int:
               continue

            seq_start = adjusted_start_int // (10 ** (digits_in_start // 2))
            seq_end = adjusted_end_int // (10 ** (digits_in_end // 2))
            # Process each range (start, end)
            first = None
            end = None
            for x in range(seq_start, seq_end + 1):
                candidate = x * (10 ** (digits_in_start // 2)) + x
                if adjusted_start_int <= candidate <= adjusted_end_int:
                    if first is None:
                        first = candidate
                    end = candidate
                    solution += candidate
            print(f"Range: {start_int} ({adjusted_start_int}) to {end_int} ({adjusted_end_int}) {seq_start} -> {seq_end}. {first} -> {end}")
    print(solution)

if __name__ == "__main__":
    main()
