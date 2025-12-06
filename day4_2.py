import sys

def get_accessible(grid):
    def get_number_neighbors(current_grid, row, col):
        offsets = [-1, 0, 1]
        neighbors = 0
        for row_offset in offsets:
            neighbor_row = row + row_offset
            if neighbor_row < 0 or neighbor_row >= len(current_grid):
                continue
            for col_offset in offsets:
                neighbor_col = col + col_offset
                if neighbor_col < 0 or neighbor_col >= len(current_grid[0]):
                    continue
                if row_offset == 0 and col_offset == 0:
                    continue
                if current_grid[neighbor_row][neighbor_col] == '@':
                    neighbors += 1
        return neighbors

    accessible = 0

    current_grid = grid
    changed = True
    while changed:
        changed = False
        new_grid = []
        for row in range(len(current_grid)):
            new_grid.append(list())
            for col in range(len(current_grid[0])):
                if current_grid[row][col] != "@":
                    print(".", end="")
                    new_grid[-1].append('.')
                    continue
                if get_number_neighbors(current_grid, row, col) < 4:
                    accessible += 1
                    changed = True
                    print("x", end = "")
                    new_grid[-1].append('.')
                else:
                    print(grid[row][col], end = "")
                    new_grid[-1].append('@')
            print()
        current_grid = new_grid
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
