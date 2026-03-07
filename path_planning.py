import heapq

def astar(start, goal, grid):

    rows, cols = len(grid), len(grid[0])

    open_set = []
    heapq.heappush(open_set,(0,start))

    came_from = {}
    cost_so_far = {start:0}

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    while open_set:

        _,current = heapq.heappop(open_set)

        if current == goal:
            break

        for dx,dy in directions:

            nx = current[0]+dx
            ny = current[1]+dy

            if 0<=nx<rows and 0<=ny<cols:

                new_cost = cost_so_far[current]+1

                if (nx,ny) not in cost_so_far:

                    cost_so_far[(nx,ny)] = new_cost
                    priority = new_cost

                    heapq.heappush(open_set,(priority,(nx,ny)))
                    came_from[(nx,ny)] = current

    path=[]
    node=goal

    while node!=start:

        path.append(node)
        node=came_from[node]

    path.append(start)
    path.reverse()

    return path