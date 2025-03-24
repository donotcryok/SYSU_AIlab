def ReverseKeyValue(dict1):
    """
    :param dict1: dict
    :return: dict
    """
    dict = {}
    for key, value in dict1.items():
        dict[value] = key
    return dict
dict1 = {'a': 1, 'b': 2, 'c': 3}
print(ReverseKeyValue(dict1))  # {1: 'a', 2: 'b', 3: 'c'}