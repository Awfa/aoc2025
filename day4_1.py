import sys

def get_accessible(grid):
    def get_number_neighbors(row, col):
        offsets = [-1, 0, 1]
        neighbors = 0
        for row_offset in offsets:
            neighbor_row = row + row_offset
            if neighbor_row < 0 or neighbor_row >= len(grid):
                continue
            for col_offset in offsets:
                neighbor_col = col + col_offset
                if neighbor_col < 0 or neighbor_col >= len(grid[0]):
                    continue
                if row_offset == 0 and col_offset == 0:
                    continue
                if grid[neighbor_row][neighbor_col] == '@':
                    neighbors += 1
        return neighbors
    accessible = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != "@":
                print(".", end="")
                continue
            if get_number_neighbors(row, col) < 4:
                accessible += 1
                print("x", end = "")
            else:
                print(grid[row][col], end = "")
        print()
    return accessible

def main():
    grid = []
    for line in sys.stdin:
        line = line.strip()
        if len(line) > 0:
            grid.append(list(line))
    solution = get_accessible(grid)
    print(solution)

if __name__ == "__main__":
    main()
