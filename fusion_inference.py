import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True, help='inference image path')
parser.add_argument('--config_path', required=True, help='config file')
parser.add_argument('--ckpt_path', required=True, help='checkpoint')
parser.add_argument('--ann', required=True, help='gt')
parser.add_argument('--out', help='output json')
args = parser.parse_args()

import os
import torch, mmcv, hashlib, json
import os.path as osp
import cv2

from mmdet.apis.inference import init_detector, inference_detector

submit_cate = {'knife': 0, 'scissors': 1, 'lighter': 2, 'zippooil': 3, 'pressure': 4,
               'slingshot': 5, 'handcuffs': 6, 'nailpolish': 7, 'powerbank': 8, 'firecrackers': 9}


def train_id_cate():
    cont = json.load(open(args.ann))
    train_cate = {cinfo['id']: cinfo['name'] for cinfo in cont['categories']}
    return train_cate


def mmdet_inference(config, ckpt):
    train_cate = train_id_cate()
    model = init_detector(config, ckpt)
    img_name_list = os.listdir(args.images)
    img_name_list.sort()
    tot_imgs = len(img_name_list)
    print("model:{},total {} images to inference".format(config, tot_imgs))

    probar = mmcv.ProgressBar(tot_imgs)
    all_img_bboxs_res = []

    for img_name in img_name_list:
        img_bbox_res = {}
        img_path = osp.join(args.images, img_name)
        img = cv2.imread(img_path, -1)
        img_w, img_h = img.shape[1], img.shape[0]
        result = inference_detector(model, img_path)
        bboxs_list = []
        scores_list = []
        labels_list = []
        for i, cont in enumerate(result):
            cont = cont.tolist()
            if cont:
                for data in cont:
                    label = i
                    submit_label = submit_cate[train_cate[label]]
                    data[0] = data[0] / img_w
                    data[2] = data[2] / img_w
                    data[1] = data[1] / img_h
                    data[3] = data[3] / img_h
                    bboxs_list.append(data[:4])
                    scores_list.append(data[-1])
                    labels_list.append(submit_label)
        img_bbox_res['img'] = img_name
        img_bbox_res['bboxs'] = bboxs_list
        img_bbox_res['scores'] = scores_list
        img_bbox_res['labels'] = labels_list
        all_img_bboxs_res.append(img_bbox_res)
        probar.update()
    return all_img_bboxs_res


def all_info_fusion():
    config_list = os.listdir(args.config_path)
    fusion_info = []
    for model in config_list:
        print(model)
        model_name = model[:-3]
        model_config = osp.join(args.config_path, model)
        model_ckpt = osp.join(args.ckpt_path, model_name + '.pth')
        all_img_bboxs_res = mmdet_inference(model_config, model_ckpt)
        fusion_info.append(all_img_bboxs_res)
    return fusion_info


def fusion_info_deal(fusion_info):
    img_num = [len(info) for info in fusion_info]
    print(img_num)
    assert len(set(img_num)) != 1
    all_img_fusion_info = []
    for id in range(img_num[0]):
        img_bboxs_res = {}
        fusion_bboxs_list = []
        fusion_scores_list = []
        fusion_labels_list = []
        for info in fusion_info:
            img = info[id]['img']
            fusion_bboxs_list.append(info['id']['bboxs'])
            fusion_scores_list.append(info['id']['scores'])
            fusion_labels_list.append(info['id']['labels'])
        img_bboxs_res['img'] = img
        img_bboxs_res['bboxs'] = fusion_bboxs_list
        img_bboxs_res['scores'] = fusion_scores_list
        img_bboxs_res['labels'] = fusion_labels_list
        all_img_fusion_info.append(img_bboxs_res)
    return all_img_fusion_info


if __name__ == '__main__':
    fusion_info = all_info_fusion()
    all_img_fusion_info = fusion_info_deal(fusion_info)
    json.dump(all_img_fusion_info, open(args.out), 'w')
    print('finish deal img num:{}'.format(len(all_img_fusion_info)))
