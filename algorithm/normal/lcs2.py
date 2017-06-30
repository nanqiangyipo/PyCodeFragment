def getcomlen(firststr, secondstr):
    comlen = 0
    while firststr and secondstr:
        if firststr[0] == secondstr[0]:
            comlen += 1
            firststr = firststr[1:]
            secondstr = secondstr[1:]
        else:
            break
    return comlen

def lcs_base(input_x, input_y):
    max_common_len = 0
    common_index = 0
    for xtemp in range(0, len(input_x)):
        for ytemp in range(0, len(input_y)):
            com_temp = getcomlen(input_x[xtemp: len(input_x)], input_y[ytemp: len(input_y)])
            if com_temp > max_common_len:
                max_common_len = com_temp
                common_index = xtemp

    print('公共子串的长度是：%s' % max_common_len)
    print('最长公共子串是：%s' % input_x[common_index:common_index + max_common_len])


if __name__ == '__main__':
    lcs_base('d11zabcdeabdcdbbcd', 'cd11yabcdefaaa')