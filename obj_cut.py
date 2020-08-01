import argparse
import json
from pycocotools.coco import COCO
import os
from cv2 import imread, imwrite
import random
import cv2
import os.path as osp
import json_info
parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True, help='a list of images to be added on bboxes')
parser.add_argument('--json', required=True, help='annotation json file in coco format')
parser.add_argument('--out_json', required=True, help='out random bbox padding info')
parser.add_argument('--out_img', metavar='REQUIRED',
                    required=True, help='output directory to put all bboxed images')
args = parser.parse_args()


def random_pad(bbox, random_rotios):
    bbox[2] = (1+2*random_rotios) * bbox[2]
    bbox[3] = (1+2*random_rotios) * bbox[3]
    bbox[0] = bbox[0]-random_rotios * bbox[2]
    bbox[1] = bbox[1]-random_rotios * bbox[3]
    return bbox

def cate_num_cont():
    data = COCO(args.json)
    json_info.json_info_print(data)


def cut_obj_save(json_file):
    data = COCO(json_file)
    cats = data.getCatIds()
    cats_info = data.loadCats(ids=cats)
    '''
        for cat in cats_info:
        out_path = os.path.join(args.out, cat['name'])
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    '''
    bbox_id = data.getAnnIds()
    bboxs_info = data.loadAnns(ids=bbox_id)
    bbox_random_info = []
    for i, bbox_info in enumerate(bboxs_info):
        img_name_info = data.loadImgs(ids=bbox_info['image_id'])
        img_name = img_name_info[0]["file_name"]
        img_path = os.path.join(args.images, img_name)
        bbox = bbox_info['bbox']  # x,y,w,h
        category = data.loadCats(ids=bbox_info['category_id'])
        if not os.path.exists(img_path):
            continue
        if bbox == 0 or bbox_info['area'] < 20:
            print("small bbox id:{}".format(bbox_info['id']))
            continue
        img = imread(img_path, -1)
        img_h, img_w, img_c = img.shape
        random_rotios = random.uniform(0.1, 0.2)
        bbox = random_pad(bbox, random_rotios)
        # constant=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=0)
        top_size = 0 if bbox[1] > 0 else abs(bbox[1])
        bottom_size = 0 if bbox[1] + bbox[3] < img_h else abs(bbox[1] + bbox[3] - img_h)
        left_size = 0 if bbox[0] > 0 else abs(bbox[0])
        right_size = 0 if bbox[0] + bbox[2] < img_w else abs(bbox[0] + bbox[2] - img_w)

        left_top = [max(0, bbox[0]), max(0, bbox[1])]
        right_bot = [min(bbox[0] + bbox[2], img.shape[1]), min(bbox[1] + bbox[3], img.shape[0])]
        if left_top[0] < right_bot[0] and left_top[1] < right_bot[1]:
            cut_img = img[int(left_top[1]):int(right_bot[1]), int(left_top[0]):int(right_bot[0])]
        else:
            print("err bbox id:{}".format(bbox_info['id']))
            print("img_name_info:{} img size:{} bbox:{}".format(img_name_info, img.shape, bbox))
            continue
        cut_img_pad = cv2.copyMakeBorder(cut_img, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                         borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
        cut_img_name = category[0]['name'] + str(bbox_info['id']) + ".jpeg"
        # save_path = os.path.join(args.out, category[0]['name'])

        print(cut_img_name)
        ann = {'id': i, 'cate_name': category[0]['name'], 'cate_id': bbox_info['category_id'],
               'random_size': random_rotios, 'img_name': cut_img_name}
        if not os.path.exists(args.out_img):
            os.makedirs(args.out_img)
        imwrite_path = os.path.join(args.out_img, cut_img_name)
        imwrite(imwrite_path, cut_img_pad)
        bbox_random_info.append(ann)
        if i % 100 == 0:
            print("finish bbox:{}".format(i))
    json.dump(bbox_random_info, open(args.out_json, 'w'))
    print("finish all bboxs")


def main():
    cate_num_cont()
    cut_obj_save(args.json)


if __name__ == '__main__':
    main()
