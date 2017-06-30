# import the necessary packages
import numpy as np


#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    """
    本质上来说我们需要一个算法，能解决从一堆候选框中挑选出最好的框。“挑选”有两个关键的过程，即选择与丢弃。从候选集中怎么选择一个最优的，怎么排除那些和最优相似的。极大值抑制就定义了这么一套规则。
    选择最优，本实现是选择候选框中右下角坐标最大的框认为是最好的框；在一些应用中，可能每一个框还有一个与之对应的置信度指标，那可以优先选择置信度最大的框作为最好的框，这就是选择策略。
    丢弃，丢弃是指丢弃和已选的最优框相似度极高的框，换种说法叫重合度过高的框。相似度计算方法是最优框与候选集中的框的交叉面积与候选集框面积的比值，丢弃是从候选集中丢掉比值大于设定阈值的所有框。
    重复以上两个过程，直到候选集中没有框。
    :param boxes: it is in the form of (startX, startY, endX, endY)
    :param overlapThresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
    return boxes[pick]