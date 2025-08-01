import pandas as pd


def check(path):
    """
    :param path:
    :return: is_true,error_message
    """
    # 错误参数 is_true
    # 1表示格式正确，0表示错误
    is_true = 1

    # 报错信息 error_message
    error_message = []

    # 读取样例数据文件
    sample = pd.read_excel('static/cluster/sample.xlsx')
    # 读取待测数据文件
    data = pd.read_excel(path)
    # 导出样例表头
    sample_head = list(sample.keys())
    del sample_head[0]
    # 导出待测数据表头
    data_head = list(data.keys())
    # 检查表头格式是否正确
    flag = 0
    error_list = []
    for i in sample_head:
        if i in data_head:
            flag += 1
        else:
            error_list.append(i)

    if flag != len(sample_head):
        is_true = 0
        error_message.append("缺少数据列：" + ','.join(error_list))
        return is_true, ';'.join(error_message)
    zyh = list(data['住院号'])
    xb = list(data['性别'])
    ID = list(data['ID'])
    nl = list(data['年龄'])
    xx = list(data['血型'])
    jzs = list(data['家族史'])
    gms = list(data['过敏史'])
    yjs = list(data['饮酒史'])
    xys = list(data['吸烟史'])
    zyts = list(data['住院天数'])
    sw = list(data['死亡'])
    bah = list(data['病案号'])
    zz = list(data['症状'])
    zy = list(data['中药'])

    # 判断住院号
    # 判断住院号是否与病案号相等
    if zyh != bah:
        is_true = 0
        error_message.append('住院号与病案号不相等')
    # 判断住院号是否为数字
    for i in zyh:
        if type(i) != int:
            is_true = 0
            error_message.append('住院号必须为整数且不能为空')
        else:
            if i < 0:
                is_true = 0
                error_message.append('住院号必须大于零')

    # 判断性别是否错误
    if list(set(xb)) != [1, 2]:
        is_true = 0
        error_message.append("性别只能使用1，2表示且不能为空")

    # 判断ID是否为数字
    for i in ID:
        if type(i) != int:
            is_true = 0
            error_message.append('ID必须为数字且不能为空')

    # 判断年龄
    for z in nl:
        if type(z) != int:
            is_true = 0
            error_message.append('年龄必须为数字且不能为空')
        else:
            if z <= 0:
                is_true = 0
                error_message.append('年龄必须大于零')

    # 判断血型
    for i in xx:
        if i not in [-1, 0, 1, 2, 3, 4]:
            is_true = 0
            error_message.append('血型使用-1，0，1，2，3，4表示且不能为空')

    # 判断家族史
    for i in jzs:
        if i not in [0, 1]:
            is_true = 0
            error_message.append('家族史使用0，1表示且不能为空')
    # 判断过敏史
    for i in gms:
        if i not in [0, 1]:
            is_true = 0
            error_message.append('过敏史使用0，1表示且不能为空')

    # 判断饮酒史
    for i in yjs:
        if i not in [0, 1]:
            is_true = 0
            error_message.append('饮酒史使用0，1表示且不能为空')

    # 判断吸烟史
    for i in xys:
        if i not in [0, 1]:
            is_true = 0
            error_message.append('吸烟史使用0，1表示且不能为空')

    # 判断住院天数
    for z in zyts:
        if type(z) != int:
            is_true = 0
            error_message.append('住院天数必须为整数数字且不能为空')
        else:
            if z < 0:
                is_true = 0
                error_message.append('住院天数必须大于零')

    # 判断死亡
    for i in sw:
        if i not in [0, 1]:
            is_true = 0
            error_message.append('死亡使用0，1表示且不能为空')

    # 判断病案号
    for z in bah:
        if type(z) != int:
            is_true = 0
            error_message.append('病案号必须为整数数字且不能为空')
        else:
            if z < 0:
                is_true = 0
                error_message.append('病案号必须大于零')

    # 判断症状
    for i in zz:
        if type(i) == float:
            is_true = 0
            error_message.append('症状不能为空')
        else:
            if '，' in i:
                is_true = 0
                error_message.append('症状使用英文逗号分隔')

    # 判断中药
    for i in zy:
        if '，' in str(i):
            is_true = 0
            error_message.append('中药使用英文逗号分隔')

    return is_true, ';'.join(list(set(error_message)))


# jg = check('static/cluster/1648290729test/data/data.xlsx')
# print(jg[1])
