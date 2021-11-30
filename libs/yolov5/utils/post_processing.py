import os


def check_rectangle_in_rectange(rectange1: list, rectange2: list):
    """
    # Check the rectange2 in rectange1 whether or not
    """
    if rectange1[0] < rectange2[0] < rectange2[2] < rectange1[2] and rectange1[1] < rectange2[1] < rectange2[3] < rectange1[3]:
        return True
    return False


def eliminate_reference(predict_bbox, reference_bboxes):
    """
    # Check the predict_bbox is in reference_bboxes or not
    """

    for bbox in reference_bboxes:
        if check_rectangle_in_rectange(bbox, predict_bbox):
            return True
    return False
    