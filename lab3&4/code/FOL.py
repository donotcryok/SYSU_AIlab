from MGU import *

def generate_identifier(literal_idx, clause_idx, clause_size):
    if clause_size == 1:
        return str(clause_idx + 1)
    else:
        return str(clause_idx + 1) + chr(ord('a') + literal_idx)

# 判断两个字面量是否互补
def are_complementary(lit1, lit2):
    if not lit1 or not lit2:
        return False
    end1 = lit1.find('(')
    end2 = lit2.find('(')
    if lit1[0] == '~' and lit1[1:end1] == lit2[:end2]:
        return True
    if lit2[0] == '~' and lit2[1:end2] == lit1[:end1]:
        return True
    return False

# 对两个子句进行归结，生成新的子句
def resolve_clauses(clause1, clause2, idx1, idx2):
    new_clause = list(clause1) + list(clause2)
    new_clause.remove(clause1[idx1])
    new_clause.remove(clause2[idx2])
    new_clause = list(dict.fromkeys(new_clause)) 
    return tuple(new_clause)

# 生成归结序列的表示形式
def format_resolution_sequence(new_clause, id1, id2, substitutions):
    if not substitutions:
        seq = 'R[' + id1 + ',' + id2 + ']='
    else:
        seq = 'R[' + id1 + ',' + id2 + ']'
        for key, value in substitutions.items():
            seq += '{' + str(key) + '=' + str(value) + '}'
        seq += '='
    seq += str(new_clause)
    return seq

# 对子句中的字面量应用替换映射
def substitute_clause(clause, substitutions):
    new_clause = []
    for literal in clause:
        new_clause.append(apply_replace(literal, substitutions))  # 替换 apply_mapping 为 apply_replace
    return tuple(new_clause)

# 归结过程
def resolution(KB):
    all_clauses = list(KB)
    support_list = [all_clauses[-1]]
    result = []
    processed_pairs = set()
    while True:
        new_clauses = []
        for i in range(len(all_clauses)):
            for j in range(i + 1, len(all_clauses)):
                if i == j:
                    continue
                clause1, clause2 = all_clauses[i], all_clauses[j]
                if (clause1, clause2) in processed_pairs:
                    continue
                if clause2 not in support_list and clause1 not in support_list:
                    continue
                for lit_idx1 in range(len(clause1)):
                    for lit_idx2 in range(len(clause2)):
                        lit1, lit2 = clause1[lit_idx1], clause2[lit_idx2]
                        if not are_complementary(lit1, lit2):
                            continue
                        lit1_clean = lit1.replace('~', '')
                        lit2_clean = lit2.replace('~', '')
                        mgu_dict = MGU([lit1_clean], [lit2_clean])
                        if mgu_dict is None:
                            continue
                        clause1_sub = substitute_clause(clause1, mgu_dict)
                        clause2_sub = substitute_clause(clause2, mgu_dict)
                        new_clause = resolve_clauses(clause1_sub, clause2_sub, lit_idx1, lit_idx2)
                        if new_clause in all_clauses or new_clause in new_clauses:
                            continue
                        processed_pairs.add((clause1, clause2))
                        id1 = generate_identifier(lit_idx1, i, len(clause1))
                        id2 = generate_identifier(lit_idx2, j, len(clause2))
                        seq = format_resolution_sequence(new_clause, id1, id2, mgu_dict)
                        result.append(seq)
                        new_clauses.append(new_clause)
                        if new_clause == ():
                            return result
        all_clauses.extend(new_clauses)
        support_list.extend(new_clauses)

#更新步骤编号
def update_num(num, steps, useful_steps, init_size):
    if num <= init_size:
        return num
    parent_seq = steps[num - 1]
    start = parent_seq.find('(')
    for i in range(init_size, len(useful_steps)):
        begin = useful_steps[i].find('(')
        if useful_steps[i][begin:] == parent_seq[start:]:
            return i + 1

#提取父步骤编号
def extract_parents(seq):
    start = seq.find('[')
    end = seq.find(']')
    nums = seq[start + 1:end].split(',')
    parent1 = int(''.join(c for c in nums[0] if not c.isalpha()))
    parent2 = int(''.join(c for c in nums[1] if not c.isalpha()))
    return parent1, parent2

#重新分配步骤编号
def reassign_sequence(seq, old_num1, old_num2, new_num1, new_num2):
    pos1 = seq.find(old_num1)
    end1 = pos1 + len(old_num1)
    seq = seq[:pos1] + new_num1 + seq[end1:]
    pos2 = seq.find(old_num2, pos1 + len(new_num1))
    end2 = pos2 + len(old_num2)
    seq = seq[:pos2] + new_num2 + seq[end2:]
    return seq

def simp_steps(steps, init_size): 
    useful = []
    queue = [len(steps)]
    visited = set()
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        useful.append(steps[cur - 1])
        p1, p2 = extract_parents(steps[cur - 1])
        if p1 > init_size:
            queue.append(p1)
        if p2 > init_size:
            queue.append(p2)
    useful.reverse()
    useful_steps = steps[0:init_size] + useful
    for i in range(init_size, len(useful_steps)):
        p1, p2 = extract_parents(useful_steps[i])
        new_num1 = str(update_num(p1, steps, useful_steps, init_size))
        new_num2 = str(update_num(p2, steps, useful_steps, init_size))
        useful_steps[i] = reassign_sequence(useful_steps[i], str(p1), str(p2), new_num1, new_num2)
    return useful_steps

def solve(KB):
    resolution_steps = list(KB.copy()) + resolution(KB)
    resolution_steps = simp_steps(resolution_steps, len(KB))  # updated call
    return resolution_steps

if __name__ == '__main__':
    KB1 = [('A(tony)',), ('A(mike)',), ('A(john)',), ('L(tony,rain)',), ('L(tony,snow)',), ('~A(x)', 'S(x)', 'C(x)'), ('~C(y)', '~L(y,rain)'), ('L(z,snow)', '~S(z)'), ('~L(tony,u)', '~L(mike,u)'), ('L(tony,v)', 'L(mike,v)'), ('~A(w)', '~C(w)', 'S(w)')]
    KB2 = [('On(tony,mike)',), ('On(mike,john)',), ('Green(tony)',), ('~Green(john)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')]
    KB3 = [
    ('On(tony,mike)',),
    ('On(mike,john)',),
    ('Green(tony)',),
    ('~Green(john)',),
    ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')
]
    KB4 = [
    ('F(f(a),g(b))',),
    ('G(h(c),k(d))',),
    ('~F(x,y)', '~G(u,v)', 'S(x,u)'),
    ('~S(w,z)', 'T(w,z)'),
    ('~T(f(a),h(c))',),
    ('S(f(a),h(c))',)
]
    KB5 = [
    ('On(tony,mike)',),
    ('On(mike,john)',),
    ('Green(tony)',),
    ('~Green(john)',),
    ('~On(xx,yy)', '~Green(xx)', 'Green(yy)'),
    ('~On(tony,z)', '~On(mike,z)', 'Green(z)')
]
    res=solve(KB1)#自行选择KB1~KB5
    for i in range(len(res)):
        print(f'{i+1} {res[i]}')