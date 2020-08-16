import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--json', required=True, help='fusion inference res')
parser.add_argument('--iou', required=True, help='iou', type=float)
parser.add_argument('--out', required=True, help='wbf res')
parser.add_argument('--img', required=True, help='img path')
parser.add_argument('--mode', default = 0,help='fusion ways,0:wbf,1:nms',type=int)

args = parser.parse_args()
import json, os
from ensemble_boxes import *


def wbf(iou_thr):
    json_info = json.load(open(args.json))
    weights = [1, 1, 1]
    all_img_wdf_res = []
    for res in json_info:
        wbf_res = {}
        boxes_list = res['bboxs']
        scores_list = res['scores']
        labels_list = res['labels']
        if args.mode == 0:
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                          iou_thr=iou_thr, skip_box_thr=0.0)
        else:
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        # print(scores)
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

        # print(scores)
        wbf_res['scores'] = [max(score, 0.0011) for score in scores]
        all_img_wdf_res.append(wbf_res)
    return all_img_wdf_res


def main():
    all_img_wbf_res = wbf(args.iou)
    all_sub_result = []
    assert args.mode == 0 or args.mode == 1
    # print(all_img_wbf_res)
    for wbf_res in all_img_wbf_res:
        sub_result = [[] for i in range(10)]
        wbf_res['bboxs'] = wbf_res['bboxs'].tolist()
        # wbf_res['scores'] = wbf_res['scores'].tolist()
        wbf_res['labels'] = wbf_res['labels'].tolist()
        for i, cont in enumerate(wbf_res['labels']):
            box_score_concat = wbf_res['bboxs'][i] + [wbf_res['scores'][i]]
            sub_result[int(cont)].append(box_score_concat)
        all_sub_result.append(sub_result)

    json.dump(all_sub_result, open(args.out, 'w'))
    print('model fusion finish,mode:{}'.format(args.mode))

if __name__ == '__main__':
    main()
