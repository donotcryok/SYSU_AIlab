def ResolutionProp(KB):
    # 初始化
    sentences = list(KB)  # 使用列表来保持顺序
    steps = []
    
    # 输出初始子句集
    for i, sentence in enumerate(sentences, 1):
        step_str = f"{i} {format_sentence(sentence)}"
        steps.append(step_str)
        
    # 准备归结
    new_sentences = sentences.copy()
    used_pairs = set() 
    step_count = len(sentences) + 1
    
    newly_added_indices = []
    
    # 核心：归结推理(优先使用新生成的子句)
    while True:
        added = False
        
        pairs_to_try = []
        
        # 如果有新添加的子句，优先使用它们
        if newly_added_indices:
            for new_idx in newly_added_indices:
                for old_idx in range(len(new_sentences)):
                    if new_idx != old_idx and (new_idx, old_idx) not in used_pairs and (old_idx, new_idx) not in used_pairs:
                        pairs_to_try.append((new_idx, old_idx))
        # 否则考虑所有未归结的对子句
        if not pairs_to_try:
            n = len(new_sentences)
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in used_pairs:
                        pairs_to_try.append((i, j))
        
        newly_added_indices = []
        
        for i, j in pairs_to_try:
            sentence_i = new_sentences[i]
            sentence_j = new_sentences[j]
            resolvent, i_lit_idx, j_lit_idx = resolve(sentence_i, sentence_j)
            
            if resolvent is not None:
                used_pairs.add((i, j))
                
                if resolvent not in new_sentences:
                    new_sentences.append(resolvent)
                    newly_added_indices.append(len(new_sentences) - 1)
                    
                    i_index = i + 1
                    j_index = j + 1
                    
                    i_suffix = ""
                    j_suffix = ""
                    
                    # 给包含多个公式的子句添加字母索引
                    if len(sentence_i) > 1 and i_lit_idx is not None:
                        i_suffix = chr(97 + i_lit_idx)
                    if len(sentence_j) > 1 and j_lit_idx is not None:
                        j_suffix = chr(97 + j_lit_idx)
                    
                    # 输出
                    step_str = f"{step_count} R[{i_index}{i_suffix},{j_index}{j_suffix}]={format_resolvent(resolvent)}"
                    steps.append(step_str)
                    step_count += 1
                    added = True
                    
                    # 如果得到空子句，说明归结成功
                    if resolvent == ():
                        return steps
                    
                    # 否则重新开始
                    break
        
        # 如果没有新的子句被添加，说明归结已经完成
        if not added:
            break
    return steps

# 归结
def resolve(sentence1, sentence2):
    for i, lit1 in enumerate(sentence1):
        for j, lit2 in enumerate(sentence2):
            if is_complement(lit1, lit2):
                new_sentence = tuple(sorted(set(sentence1 + sentence2) - {lit1, lit2}))
                return new_sentence, i, j
    
    return None, None, None

# 是否互补
def is_complement(lit1, lit2):
    if lit1.startswith('~') and lit1[1:] == lit2:
        return True
    if lit2.startswith('~') and lit2[1:] == lit1:
        return True
    return False

# 格式化
def format_sentence(sentence):
    if not sentence:
        return "()"
    return "(" + ",".join(sentence) + ",)"

def format_resolvent(sentence):
    if not sentence:
        return "()"
    return "(" + ",".join(sentence) + ",)"

# 测试
def test():
    KB = [('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)]
    
    steps = ResolutionProp(KB)
    
    for step in steps:
        print(step)
    

if __name__ == "__main__":
    test()
