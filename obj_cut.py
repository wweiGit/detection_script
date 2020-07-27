import argparse
import json
from pycocotools.coco import COCO
import os
from cv2 import imread, imwrite


parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True, help='a list of images to be added on bboxes')
parser.add_argument('--json',required=True, help='annotation json file in coco format')
parser.add_argument('--out', action='store', dest='out', metavar='REQUIRED',
                    required=True, help='output directory to put all bboxed images')
args = parser.parse_args()

def cut_obj_save(json_file):
    data = COCO(json_file)
    cats = data.getCatIds()
    cats_info = data.loadCats(ids=cats)
    for cat in cats_info:
        out_path = os.path.join(args.out,cat['name'])
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    bbox_id = data.getAnnIds()
    bboxs_info =data.loadAnns(ids=bbox_id)

    for i,bbox_info in enumerate(bboxs_info):
        img_name_info = data.loadImgs(ids=bbox_info['image_id'])
        img_name = img_name_info[0]["file_name"]
        img_path = os.path.join(args.images,img_name)
        bbox = bbox_info['bbox']  # x,y,w,h
        category = data.loadCats(ids=bbox_info['category_id'])
        if not os.path.exists(img_path):
            continue
        if bbox==0 or bbox_info['area']<30:
            print("small bbox id:{}".format(bbox_info['id']))
            continue
        img = imread(img_path, -1)
        left_top = [bbox[0],bbox[1]]
        right_bot = [min(bbox[0]+bbox[2],img.shape[1]),min(bbox[1]+bbox[3],img.shape[0])]

        if left_top[0]<right_bot[0] or left_top[1]<right_bot[1]:
            cut_img = img[left_top[1]:right_bot[1],left_top[0]:right_bot[0]]
        else:
            print("err bbox id:{}".format(bbox_info['id']))
            print("img_name_info:{} img size:{} bbox:{}".format(img_name_info,img.shape,bbox))
            continue

        cut_img_name = img_name[:-5] + "_" +category[0]['name']+str(bbox_info['id'])+".jpeg"

        save_path = os.path.join(args.out, category[0]['name'])

        imwrite_path = os.path.join(save_path, cut_img_name)

        imwrite(imwrite_path, cut_img)
        if i%100==0:
            print("finish bbox:{}".format(i))
    print("finish all bboxs")


def main():
    cut_obj_save(args.json)


if __name__ == '__main__':
    main()



