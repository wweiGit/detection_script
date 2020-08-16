import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--json', required=True, help='fusion inference res')
parser.add_argument('--iou', required=True, help='iou')
parser.add_argument('--out', required=True, help='wbf res')
parser.add_argument('--img', required=True, help='img path')
args = parser.parse_args()
import json, os
from ensemble_boxes import *


def wbf(iou_thr=0.55):
    json_info = json.load(open(args.json))
    weights = [1, 1, 1]
    all_img_wdf_res = []
    for res in json_info:
        wbf_res = {}
        boxes_list = res['bboxs']
        scores_list = res['scores']
        labels_list = res['labels']
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=0.0)
        #print(scores)
        wbf_res['img'] = res['img']
        img = cv2.imread(os.path.join(args.img, wbf_res['img']))
        img_h, img_w, _ = img.shape
        # print(boxes)
        for box in boxes:
            box[0] = box[0] * img_w
            box[2] = box[2] * img_w
            box[1] = box[1] * img_h
            box[3] = box[3] * img_h
        # print(boxes)
        wbf_res['bboxs'] = boxes
        wbf_res['labels'] = labels
        wbf_res['scores'] = scores
        all_img_wdf_res.append(wbf_res)
    return all_img_wdf_res


def main():
    all_img_wbf_res = wbf(0.55)
    all_sub_result = []
    print(all_img_wbf_res)
    for wbf_res in all_img_wbf_res:
        sub_result = [[] for i in range(10)]
        wbf_res['bboxs'] = wbf_res['bboxs'].tolist()
        wbf_res['scores'] = wbf_res['scores'].tolist()
        wbf_res['labels'] = wbf_res['labels'].tolist()
        for i, cont in enumerate(wbf_res['labels']):
            box_score_concat = wbf_res['bboxs'][i] + [wbf_res['scores'][i]]
            sub_result[int(cont)].append(box_score_concat)
        all_sub_result.append(sub_result)

    json.dump(all_sub_result, open(args.out, 'w'))


if __name__ == '__main__':
    main()
