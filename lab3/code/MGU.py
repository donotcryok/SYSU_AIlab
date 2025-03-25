# 解析原子公式，得到谓词和参数列表
# 例如：parse_atom('P(a,b,c)') -> ('P', ['a', 'b', 'c'])
def parse_atom(atom):
    bracket = atom.find('(')
    return atom[:bracket], parse_terms(atom[bracket + 1:-1])

# 解析参数列表，特别注意嵌套情况的处理
# 例如：parse_terms('a,b(c,d(e,f))') -> ['a', 'b(c,d(e,f))']
def parse_terms(params_str):
    terms, current_param, level = [], "", 0
    for char in params_str:
        if char == ',' and level == 0:#level处理嵌套
            terms.append(current_param.strip())
            current_param = ""
        else:
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            current_param += char
    if current_param:
        terms.append(current_param.strip())
    return terms

# 判断是否为变量（恰好两个连续相同的小写字母，非复合项）
def is_variable(term):
    # 变量必须是恰好两个连续相同的小写字母，并且不含括号
    return (term and len(term) == 2 and 
            term[0] == term[1] and 
            term[0].islower() and 
            '(' not in term)

# 判断是否为常量（非变量且非复合项）
def is_constant(term):
    # 常量是非变量且非复合项
    return term and not is_variable(term) and '(' not in term

# 判断是否为复合项（包含括号）
def is_compound_term(term):
    return '(' in term and ')' in term

# 递归应用替换规则
def apply_replace(term, replace):
    if is_variable(term):
        return replace.get(term, term)
    if is_compound_term(term):
        func, params = parse_atom(term)
        return f"{func}({','.join(apply_replace(p, replace) for p in params)})"
    return term

# 检查变量是否出现在项中，避免循环替换
def check(var, term, replace):
    if var == term:
        return True
    if is_variable(term) and term in replace:
        return check(var, replace[term], replace)
    if is_compound_term(term):
        return any(check(var, p, replace) for p in parse_terms(term[term.find('(') + 1:-1]))
    return False

# 变量合一
def unify_variable(var, term, replace):
    if var in replace:
        return unify(replace[var], term, replace)
    if is_variable(term):
        # 如果term也是变量，那么两变量可以统一
        if check(var, term, replace):
            return None
        replace[var] = term
        return replace
    # 检查循环依赖
    if is_compound_term(term) and check(var, term, replace):
        return None
    # 变量可以替换为常量或复合项
    replace[var] = term
    return replace

# 最一般合一（MGU）核心逻辑
def unify(x, y, replace):
    if replace is None:
        return None
    if x == y:
        return replace
    if is_variable(x):
        return unify_variable(x, y, replace)
    if is_variable(y):
        return unify_variable(y, x, replace)
    if is_compound_term(x) and is_compound_term(y):
        xf, xp = parse_atom(x)
        yf, yp = parse_atom(y)
        if xf != yf or len(xp) != len(yp):
            return None
        for i in range(len(xp)):
            replace = unify(xp[i], yp[i], replace)
            if replace is None:
                return None
        return replace
    # 如果都是常量且不相等，无法合一
    return None if x != y else replace

# 计算两个原子公式的最一般合一（MGU）
def MGU(atom1, atom2):
    # 检查是否是字符串类型，如果是列表，则取第一个元素
    if isinstance(atom1, list) and len(atom1) > 0:
        atom1 = atom1[0]
    if isinstance(atom2, list) and len(atom2) > 0:
        atom2 = atom2[0]
    
    # 解析原子公式
    try:
        pred1, params1 = parse_atom(atom1)
        pred2, params2 = parse_atom(atom2)
    except:
        return {}  # 解析失败时返回空字典
    
    if pred1 != pred2 or len(params1) != len(params2):
        return {}
    replace = {}
    for p1, p2 in zip(params1, params2):
        replace = unify(p1, p2, replace)
        if replace is None:
            return {}
    return replace

# 确保parse_terms可以被infer2.py导入
__all__ = ['MGU', 'parse_terms', 'parse_atom', 'apply_replace']

# 格式化替换结果
def format_replace(replace):
    return "{}" if not replace else "{" + ", ".join(f"{var}={term}" for var, term in replace.items()) + "}"

# 测试代码
if __name__ == "__main__":
    print(f"MGU('P(xx,a)', 'P(b,yy)') result is {format_replace(MGU('P(xx,a)', 'P(b,yy)'))}")
    print(f"MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))') result is {format_replace(MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))'))}")
