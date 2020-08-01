#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('images', help='Test image path.')
parser.add_argument('--config', required=True, help='Configure file')
parser.add_argument('--ckpt', required=True, help='Checkpoint')
parser.add_argument('--ann', help='Annotation json, could be a gt.')
parser.add_argument('--out', required=True, help='Output json')
parser.add_argument('--gpu', required=True, help='Choose GPU to run model on')

args = parser.parse_args()

import os
import torch, mmcv, hashlib, json
import os.path as osp
from mmdet.apis.inference import init_detector, inference_detector

train_cate = {0: 'firecrackers', 1: 'handcuffs', 2: 'knife', 3: 'lighter', 4:
    'nailpolish', 5: 'powerbank', 6: 'pressure', 7: 'scissors', 8: 'slingshot', 9: 'zippooil'}

submit_cate = {'knife': 0, 'scissors': 1, 'lighter': 2, 'zippooil': 3, 'pressure': 4,
               'slingshot': 5,'handcuffs': 6, 'nailpolish': 7,'powerbank': 8, 'firecrackers': 9}

def check_args():
    args.gpu = torch.device('cuda:{}'.format(args.gpu))
    if args.ann is not None:
        args.ann = load_annotation(args.ann)

def load_annotation(ann):
    cont = json.load(open(ann))
    iname2iid = {img['file_name']: img['id'] for img in cont['images']}
    cidx2cid = {i: cinfo['id'] for i, cinfo in enumerate(cont['categories'])}
    cidx2cname = {i: cinfo['name'] for i, cinfo in enumerate(cont['categories'])}
    return iname2iid, cidx2cid, cidx2cname

def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]

def str2md5(s):
    md5 = hashlib.md5()
    md5.update(s.encode('utf8'))
    return int(md5.hexdigest(), 16)

def main():
    check_args()
    model = init_detector(args.config, args.ckpt, args.gpu)
    img_name_list = os.listdir(args.images)
    img_name_list.sort()
    tot_imgs = len(img_name_list)
    print('Total {} images to inference'.format(tot_imgs))
    progbar = mmcv.ProgressBar(len(img_name_list))
    all_sub_result = []
    for img_name in img_name_list:
        sub_result = [[] for i in range(10)]
        img_path = osp.join(args.images,img_name)
        result = inference_detector(model, img_path)
        for i,cont in enumerate(result):
            cont = cont.tolist()
            #print("**********************************")
            #for tmp in cont:
                #print(tmp)
            cate_name = train_cate[i]
            sub_index = submit_cate[cate_name]
            #print('train_cate_id:{},sub_index:{}'.format(i,sub_index))
            sub_result[sub_index] = cont
        all_sub_result.append(sub_result)
        progbar.update()
    json.dump(all_sub_result, open(args.out, 'w'))
    print("finish!")


if __name__ == '__main__':
    main()


