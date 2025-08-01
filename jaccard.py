def jaccard(set_x, set_y):
    intersection = set_x.intersection(set_y)
    union = set_x.union(set_y)
    return round(float(len(intersection) / len(union)), 2)


def compute_label(feature_str, label_feature: list):
    new_feature = set(feature_str.split(";"))
    max_label = 0
    similarity = 0
    js = 0
    for i in label_feature:
        if jaccard(new_feature, set(i.split(','))) > similarity:
            similarity = jaccard(new_feature, set(i.split(',')))
            max_label = js
        js += 1
    return max_label
