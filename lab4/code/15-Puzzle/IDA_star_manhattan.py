import time
import heapq
from collections import deque

# 目标状态 - 便于快速查找每个数字的目标位置
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
# 预计算每个位置的目标坐标
GOAL_POSITIONS = {GOAL_STATE[i]: (i // 4, i % 4) for i in range(16)}

INITIAL_STATE_1 = [[1,2,4,8],[5,7,11,10],[13,15,0,3],[14,6,9,12]]
INITIAL_STATE_2 = [[14,10,6,0],[4,9,1,8],[2,3,5,11],[12,13,7,15]]
INITIAL_STATE_3 = [[5,1,3,4],[2,7,8,12],[9,6,11,15],[0,13,10,14]]
INITIAL_STATE_4 = [[6,10,3,15],[14,8,7,11],[5,1,0,2],[13,12,9,4]]
INITIAL_STATE_5 = [[11,3,1,7],[4,6,8,2],[15,9,10,13],[14,12,5,0]]
INITIAL_STATE_6 = [[0,5,15,14],[7,9,6,13],[1,2,12,10],[8,11,4,3]]
INITIAL_STATE_7 = [[2,1,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]

# 用于优化的方向数组和合法移动检查
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_NAMES = ["上", "下", "左", "右"]
OPPOSITE_MOVES = {0: 1, 1: 0, 2: 3, 3: 2}  # 相反方向索引

def convert_to_tuple(state_2d):
    return tuple(num for row in state_2d for num in row)

# 输出当前状态
def print_board(state):
    print('-----------')
    for i in range(0, 16, 4):
        print(' '.join(f'{state[i+j]:2d}' for j in range(4)))
    print('-----------')

def get_manhattan_distance(state):
    """计算当前状态的曼哈顿距离"""
    distance = 0
    for i, num in enumerate(state):
        if num == 0:  # 跳过空格
            continue
        # 当前位置和目标位置
        curr_row, curr_col = i // 4, i % 4
        goal_row, goal_col = GOAL_POSITIONS[num]
        # 曼哈顿距离 = 行差异 + 列差异
        distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
    return distance

# 添加线性冲突启发式函数增强
def count_linear_conflicts(state):
    """计算线性冲突，每个冲突增加2步移动"""
    conflicts = 0
    
    # 检查每一行的冲突
    for row in range(4):
        for i in range(row * 4, row * 4 + 4):
            if state[i] == 0:
                continue
            goal_row, _ = GOAL_POSITIONS[state[i]]
            if goal_row == row:  # 如果在目标行
                for j in range(i + 1, row * 4 + 4):
                    if state[j] == 0:
                        continue
                    goal_row_j, _ = GOAL_POSITIONS[state[j]]
                    if goal_row_j == row and state[i] > state[j]:  # 如果有冲突
                        conflicts += 2
    
    # 检查每一列的冲突
    for col in range(4):
        for i in range(col, 16, 4):
            if state[i] == 0:
                continue
            _, goal_col = GOAL_POSITIONS[state[i]]
            if goal_col == col:  # 如果在目标列
                for j in range(i + 4, 16, 4):
                    if state[j] == 0:
                        continue
                    _, goal_col_j = GOAL_POSITIONS[state[j]]
                    if goal_col_j == col and state[i] > state[j]:  # 如果有冲突
                        conflicts += 2
    
    return conflicts

# 增强的启发式函数
def get_enhanced_heuristic(state):
    """增强的启发式函数: 曼哈顿距离 + 线性冲突"""
    return get_manhattan_distance(state) + count_linear_conflicts(state)

def get_blank_position(state):
    return state.index(0)

# 优化的获取邻居状态函数 - 使用方向序列避免重复创建列表
def get_neighbors_with_move(state, last_move=-1):
    """获取所有可能的邻居状态，避免回到上一状态"""
    blank_idx = get_blank_position(state)
    blank_row, blank_col = blank_idx // 4, blank_idx % 4
    
    for dir_idx, (dr, dc) in enumerate(DIRECTIONS):
        # 跳过与上一步相反的移动
        if last_move != -1 and dir_idx == OPPOSITE_MOVES[last_move]:
            continue
            
        new_row, new_col = blank_row + dr, blank_col + dc
        
        # 检查是否越界
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_idx = new_row * 4 + new_col
            # 创建新状态 - 交换空格和目标位置的值
            new_state = list(state)
            new_state[blank_idx], new_state[new_idx] = new_state[new_idx], new_state[blank_idx]
            yield tuple(new_state), dir_idx

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

def parse_input():
    """解析输入的初始状态"""
    state = []
    for _ in range(4):
        row = list(map(int, input().strip().split()))
        state.extend(row)
    return tuple(state)

def is_solvable(state):
    """检查15数码问题是否有解"""
    # 计算逆序数
    inversions = 0
    state_list = list(state)
    
    for i in range(len(state_list)):
        if state_list[i] == 0:
            continue  # 跳过空格
        for j in range(i+1, len(state_list)):
            if state_list[j] == 0:
                continue  # 跳过空格
            if state_list[i] > state_list[j]:
                inversions += 1
    
    # 获取空格所在行（从上往下数，0-indexed）
    blank_row = state.index(0) // 4
    
    # 判断是否有解：奇数行空格时逆序数需是偶数，偶数行空格时逆序数需是奇数
    return (blank_row % 2 == 0 and inversions % 2 == 1) or (blank_row % 2 == 1 and inversions % 2 == 0)

def main():
    # 根据用户需要选择初始状态
    initial_states = {
        1: INITIAL_STATE_1,
        2: INITIAL_STATE_2,
        3: INITIAL_STATE_3,
        4: INITIAL_STATE_4,
        5: INITIAL_STATE_5,
        6: INITIAL_STATE_6,
        7: INITIAL_STATE_7,
    }
    #在这里可以修改为其他状态！！！！
    state_num = 1
    initial_state_2d = initial_states[state_num]
    
    initial_state = convert_to_tuple(initial_state_2d)
    
    print("\n初始状态:")
    print_board(initial_state)
    
    # 首先检查是否有解
    if not is_solvable(initial_state):
        print("\n此状态无解！检查逆序数时发现无解。")
        return
    
    print("初始状态检查有解，继续求解...")
    print(f"初始曼哈顿距离: {get_manhattan_distance(initial_state)}")
    print(f"初始增强启发值: {get_enhanced_heuristic(initial_state)}")
    
    # 开始计时
    start_time = time.time()
    
    # 使用Pattern Database启发式函数优化
    def get_pattern_database_heuristic(state):
        """使用预计算的模式数据库启发式函数"""
        # 这里简化实现，实际中可以使用更复杂的模式数据库
        h1 = get_manhattan_distance(state)
        h2 = count_linear_conflicts(state)
        
        # 为了避免过度低估，添加角度惩罚
        corner_penalty = 0
        # 检查每个角落的位置是否正确
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        corner_values = [1, 4, 13, 15]  # 对应角落应有的值
        
        for (r, c), val in zip(corners, corner_values):
            pos = r * 4 + c
            if state[pos] != val and state[pos] != 0:
                corner_penalty += 2
                
        return h1 + h2 + corner_penalty
    
    def ida_star_search(initial_state):
        """使用改进的IDA*算法解决15数码问题"""
        if initial_state == GOAL_STATE:
            return [initial_state], []
        
        # 使用优化的启发式函数
        initial_h = get_pattern_database_heuristic(initial_state)
        
        # 初始深度限制
        bound = initial_h
        max_depth = 100  # 设置最大深度
        iterations = 0
        
        print(f"初始边界值: {bound}")
        
        def dfs(path, g, bound, last_move=-1, moves=None):
            """深度优先搜索，使用迭代加深"""
            nonlocal iterations
            iterations += 1
            
            current = path[-1]
            h = get_pattern_database_heuristic(current)
            f = g + h
            
            if f > bound:
                return f, False, path, moves
                
            if current == GOAL_STATE:
                return 0, True, path, moves
            
            min_cost = float('inf')
            
            # 生成所有可能的下一步
            for next_state, move_dir in get_neighbors_with_move(current, last_move):
                # 避免循环
                if next_state in path:
                    continue
                
                # 深度搜索
                new_path = path + [next_state]
                new_moves = moves + [move_dir] if moves is not None else [move_dir]
                
                cost, found, result_path, result_moves = dfs(new_path, g+1, bound, move_dir, new_moves)
                
                if found:
                    return cost, True, result_path, result_moves
                
                if cost < min_cost:
                    min_cost = cost
            
            return min_cost, False, path, moves
        
        path = [initial_state]
        moves = []
        
        while bound < max_depth:
            print(f"搜索深度界限: {bound}")
            # 每增加一个深度限制，就进行一次搜索
            bound, found, path, moves = dfs(path, 0, bound, -1, [])
            
            # 如果找到解，或者无解，或者达到最大深度
            if found:
                # 构建移动序列
                moved_numbers = []
                for i in range(len(moves)):
                    prev_state = path[i]
                    blank_pos = get_blank_position(prev_state)
                    dr, dc = DIRECTIONS[moves[i]]
                    new_blank_row, new_blank_col = blank_pos // 4 + dr, blank_pos % 4 + dc
                    new_blank_pos = new_blank_row * 4 + new_blank_col
                    moved_numbers.append(prev_state[new_blank_pos])
                
                print(f"找到解决方案，共搜索{iterations}个节点")
                return path, moved_numbers
            
            if bound == float('inf'):
                print("搜索失败：无法找到解")
                return None, None
            
            print(f"深度{bound}搜索完成，扩展到{iterations}个节点")
            iterations = 0
        
        print(f"达到最大深度{max_depth}，搜索终止")
        return None, None
    
    try:
        # 使用改进的IDA*算法求解
        solution, moved_numbers = ida_star_search(initial_state)
        
        # 输出结果
        if solution:
            print("\n解决方案:")
            for step in range(len(solution)-1):
                print(f"步骤 {step+1}:")
                print_board(solution[step+1])
                move_description = describe_move(solution[step], solution[step+1])
                print(move_description)
                
            print(f"\n找到解决方案! 共{len(solution)-1}步")
            if moved_numbers:
                print("移动数字序列:", " -> ".join(map(str, moved_numbers)))
        else:
            print("\n无法找到解决方案! 可能是参数限制或问题太复杂。")
    
    except MemoryError:
        print("\n内存不足! 问题可能太复杂或需要更高效的算法。")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"求解时间: {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    main()
