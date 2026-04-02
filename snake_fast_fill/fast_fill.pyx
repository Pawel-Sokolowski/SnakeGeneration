# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

from libcpp.deque cimport deque
from libcpp.vector cimport vector

cdef inline int pos_id(int x, int y, int g):
    return y * g + x


cpdef int flood_fill_cpp(int start_x, int start_y,
                         int g,
                         list blocked_cells):
    """
    Ultra-fast BFS flood fill in C++.
    blocked_cells: list of (x,y) Python tuples
    """

    cdef int grid_size = g
    cdef int n = grid_size * grid_size

    # visited flags
    cdef vector[char] visited
    visited.resize(n, 0)

    # blocked mask
    cdef vector[char] blocked
    blocked.resize(n, 0)

    cdef int bx, by, bidx
    for bx, by in blocked_cells:
        if 0 <= bx < g and 0 <= by < g:
            bidx = pos_id(bx, by, g)
            blocked[bidx] = 1

    # BFS deque
    cdef deque[int] q

    cdef int sx = start_x
    cdef int sy = start_y
    cdef int sidx = pos_id(sx, sy, g)

    if blocked[sidx] == 1:
        return 0

    q.push_back(sidx)
    visited[sidx] = 1

    cdef int count = 0
    cdef int idx, x, y
    cdef int nx, ny, nidx

    while not q.empty():
        idx = q.front()
        q.pop_front()

        count += 1

        x = idx % g
        y = idx // g

        # 4 neighbors
        # dx, dy = (1,0), (-1,0), (0,1), (0,-1)
        nx = x + 1
        if nx < g:
            nidx = pos_id(nx, y, g)
            if visited[nidx] == 0 and blocked[nidx] == 0:
                visited[nidx] = 1
                q.push_back(nidx)

        nx = x - 1
        if nx >= 0:
            nidx = pos_id(nx, y, g)
            if visited[nidx] == 0 and blocked[nidx] == 0:
                visited[nidx] = 1
                q.push_back(nidx)

        ny = y + 1
        if ny < g:
            nidx = pos_id(x, ny, g)
            if visited[nidx] == 0 and blocked[nidx] == 0:
                visited[nidx] = 1
                q.push_back(nidx)

        ny = y - 1
        if ny >= 0:
            nidx = pos_id(x, ny, g)
            if visited[nidx] == 0 and blocked[nidx] == 0:
                visited[nidx] = 1
                q.push_back(nidx)

    return count