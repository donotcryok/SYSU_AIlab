import time
import heapq
from collections import deque

# 目标状态
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
GOAL_POSITIONS = {GOAL_STATE[i]: (i // 4, i % 4) for i in range(16)}

INITIAL_STATE_1 = [[1,2,4,8],[5,7,11,10],[13,15,0,3],[14,6,9,12]]
INITIAL_STATE_2 = [[14,10,6,0],[4,9,1,8],[2,3,5,11],[12,13,7,15]]
INITIAL_STATE_3 = [[5,1,3,4],[2,7,8,12],[9,6,11,15],[0,13,10,14]]
INITIAL_STATE_4 = [[6,10,3,15],[14,8,7,11],[5,1,0,2],[13,12,9,4]]
INITIAL_STATE_5 = [[11,3,1,7],[4,6,8,2],[15,9,10,13],[14,12,5,0]]
INITIAL_STATE_6 = [[0,5,15,14],[7,9,6,13],[1,2,12,10],[8,11,4,3]]
INITIAL_STATE_7 = [[2,1,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]

'''因为之前没有注意超算习堂的样例，一直写的是一维形式的样例，转换一下'''
def convert_to_tuple(state_2d):
    return tuple(num for row in state_2d for num in row)

#输出当前状态
def print_board(state):
    print('-----------')
    for i in range(0, 16, 4):
        print(' '.join(f'{state[i+j]:2d}' for j in range(4)))
    print('-----------')

#计算曼哈顿距离
def get_manhattan_distance(state):
    distance = 0
    for i, num in enumerate(state):
        if num != 0:  # 跳过空格
            goal_row, goal_col = GOAL_POSITIONS[num]
            curr_row, curr_col = i // 4, i % 4
            distance += abs(goal_row - curr_row) + abs(goal_col - curr_col)
    return distance

#找到空格(0)
def get_blank_position(state):
    return state.index(0)

#获取所有可能的邻居状态
def get_neighbors(state):
    neighbors = []
    blank_idx = get_blank_position(state)
    blank_row, blank_col = blank_idx // 4, blank_idx % 4
    
    # 移动方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        new_row, new_col = blank_row + dr, blank_col + dc
        
        # 检查是否越界
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_idx = new_row * 4 + new_col
            # 创建新状态 - 交换空格和目标位置的值
            new_state = list(state)
            new_state[blank_idx], new_state[new_idx] = new_state[new_idx], new_state[blank_idx]
            neighbors.append(tuple(new_state))
            
    return neighbors

#描述从一个状态到另一个状态的移动方向
def describe_move(prev_state, next_state):
    prev_blank = get_blank_position(prev_state)
    next_blank = get_blank_position(next_state)
    
    # 计算空格的行和列
    prev_row, prev_col = prev_blank // 4, prev_blank % 4
    next_row, next_col = next_blank // 4, next_blank % 4
    
    # 确定移动的数字
    moved_number = prev_state[next_blank]
    
    # 确定移动方向
    if next_row < prev_row:
        direction = "上"
    elif next_row > prev_row:
        direction = "下"
    elif next_col < prev_col:
        direction = "左"
    else:
        direction = "右"
    return f"数字 {moved_number} 向{direction}移动"

def a_star_search(initial_state):
    """使用A*算法解决15数码问题"""
    if initial_state == GOAL_STATE:
        return [initial_state]
    
    # 优先队列 - (f值, 移动步数, 状态, 父状态)
    open_set = [(get_manhattan_distance(initial_state), 0, initial_state, None)]
    # 使用字典记录已访问状态的最佳路径信息
    closed_set = {}
    
    while open_set:
        # 获取f值最小的状态
        _, moves, current_state, parent = heapq.heappop(open_set)
        # 如果已经处理过此状态且有更好的路径，则跳过
        if current_state in closed_set and closed_set[current_state][0] <= moves:
            continue
        # 记录当前状态的最佳路径信息
        closed_set[current_state] = (moves, parent)
        # 检查是否达到
        if current_state == GOAL_STATE:
            # 重建路径
            path = []
            while current_state:
                path.append(current_state)
                _, current_state = closed_set.get(current_state, (0, None))
            return path[::-1]  # 反转路径
        # 探索邻居状态
        for neighbor in get_neighbors(current_state):
            new_moves = moves + 1
            # 如果已经访问过且没有更好的路径，则跳过
            if neighbor in closed_set and closed_set[neighbor][0] <= new_moves:
                continue
            # 计算f值 = g值(移动步数) + h值(曼哈顿距离)
            f_value = new_moves + get_manhattan_distance(neighbor)
            heapq.heappush(open_set, (f_value, new_moves, neighbor, current_state))
    
    return None  # 无解

def parse_input():
    state = []
    for _ in range(4):
        row = list(map(int, input().strip().split()))
        state.extend(row)
    return tuple(state)

# 检查是否有解
def is_solvable(state):
    # 计算逆序数
    inversions = 0
    for i in range(len(state)):
        if state[i] == 0:
            continue  # 跳过空格
        for j in range(i+1, len(state)):
            if state[j] == 0:
                continue  # 跳过空格
            if state[i] > state[j]:
                inversions += 1
    
    # 获取空格所在行（从下往上数）
    blank_idx = state.index(0)
    blank_row = 4 - (blank_idx // 4)  # 从下往上数的行号（1-4）
    
    # 判断是否有解
    if blank_row % 2 == 0:  # 空格在偶数行
        return inversions % 2 == 1
    else:  # 空格在奇数行
        return inversions % 2 == 0

def main():
    # 使用自定义的初始状态
    initial_state_2d = INITIAL_STATE_6  # 可以修改为任何一个初始状态
    
    initial_state = convert_to_tuple(initial_state_2d)
    
    print("\n初始状态:")
    print_board(initial_state)
    
    # 检查是否有解
    if not is_solvable(initial_state):
        print("\n此状态无解!")
        return
    
    # 开始计时
    start_time = time.time()
    
    # 使用A*算法求解
    solution = a_star_search(initial_state)
    
    # 结束计时
    end_time = time.time()
    
    # 输出结果
    if solution:
        # 记录移动的数字序列
        moved_numbers = []
        for step in range(len(solution)-1):
            print(f"\n步骤 {step+1}:")
            print_board(solution[step+1])
            # 描述这一步的移动
            move_description = describe_move(solution[step], solution[step+1])
            print(move_description)
            # 获取移动的数字
            prev_blank = get_blank_position(solution[step])
            next_blank = get_blank_position(solution[step+1])
            moved_number = solution[step][next_blank]
            moved_numbers.append(moved_number)
        
        # 输出移动数字序列
        print(f"\n找到解决方案! 共{len(solution)-1}步")
        print("\n移动顺序:", " -> ".join(map(str, moved_numbers)))
    else:
        print("\n无法解决!")
        
    print(f"求解时间: {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    main()
