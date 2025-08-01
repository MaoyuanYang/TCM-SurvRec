import pandas as pd

similarity = {}
df = pd.read_excel(".\datasets\classic_prescription.xlsx", index_col=0)

p = df.set_index("name").to_dict()["drug"]

for key in p:
    p[key] = p[key].split(",")


def jaccard(set_x: set, set_y: set) -> float:
    """
    Calculate jaccard index with 2 string sets.
    Note that the union set can't be empty!
    :param set_x: set x
    :param set_y: set y
    :return: jaccard value, float type.
    """
    intersection = set_x.intersection(set_y)
    union = set_x.union(set_y)
    return len(intersection) / len(union)



def similar(f: set):
    for key in p:
        similarity[key] = jaccard(set(p[key]), f)
    L = list(similarity.items())
    L.sort(key=lambda x: x[1], reverse=True)
    return L[0], p[L[0][0]]

# print(similar({	'甘草','当归','茯苓','白术','赤芍','丹参','白芍','川芎','黄芪','黄柏'}))
