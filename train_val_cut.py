import json
from pycocotools.coco import COCO
import argparse
parser = argparse.ArgumentParser(description='train val img cut')
parser.add_argument('--json', help='json file path')
parser.add_argument('--thr', help='train data threshold',type=float)
parser.add_argument('--out', help='train val save path',default='./')
args = parser.parse_args()


def train_cut():
    file = COCO(args.json)
    train_json = json.load(open(args.json))
    cate_id_list = file.getCatIds()

    cate_ann_thr = [int(len(file.getAnnIds(catIds=[i]))*args.thr) for i in cate_id_list]
    print(cate_ann_thr)

    cate_ann_num = {i:0 for i in cate_id_list}

    train_img = set()

    for ann in train_json['annotations']:

        if cate_ann_thr[ann['category_id']] > cate_ann_num[ann['category_id']]:
            cate_ann_num[ann['category_id']]+=1
            train_img.add(ann['image_id'])

    train_json['annotations'] = [ann for ann in train_json['annotations'] if ann['image_id'] in train_img]
    train_json['images'] =[img for img in train_json['images'] if img['id'] in train_img]
    with open('train.json','w') as f:
        f.write(json.dumps(train_json))
    return train_json,train_img


def val_cut(train_img):
    val_json = json.load(open(args.json))
    val_json['annotations'] = [ann for ann in val_json['annotations'] if ann['image_id'] not in train_img]
    val_json['images'] = [img for img in val_json['images'] if img['id'] not in train_img]

    with open('val.json','w') as f:
        f.write(json.dumps(val_json))

    print(val_json.keys())
    return val_json


def json_file_cont(json_file):
    img_cont = len([file for file in json_file['images']])
    ann_cont = len([file for file in json_file['annotations']])
    return img_cont,ann_cont

if __name__ == '__main__':
    train_json,train_img = train_cut()
    val_json = val_cut(train_img)
    print('finish train val cut!')
    train_img_cont,train_ann_cont = json_file_cont(train_json)
    test_img_cont,test_ann_cont = json_file_cont(val_json)
    all_info_img,all_info_ann = json_file_cont(json.load(open(args.json)))
    print("all_img_cont:{},all_ann_cont:{},train_img_cont:{},train_ann_cont:{},test_img_cont:{},test_ann_cont:{}".format(
        all_info_img,all_info_ann,train_img_cont, train_ann_cont,test_img_cont,test_ann_cont
    ))