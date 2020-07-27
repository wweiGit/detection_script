from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import json
json_path = '/home/wang/data/ifly_xdet/xdet_data/json/all_img_info.json'
train_json = './train.json'
test_json = './val.json'


def json_info_print(file):
    all_img_num = file.getImgIds()
    all_bbox_mum = file.getAnnIds()
    all_cat_num = file.getCatIds()
    cat_name_info = file.loadCats(ids=all_cat_num)
    cat_name = [cat_info['name'] for cat_info in cat_name_info]
    img_of_cat_cont = [len(file.getImgIds(catIds=[i])) for i in all_cat_num]
    bbox_of_cat_cont = [len(file.getAnnIds(catIds=[i])) for i in all_cat_num]
    img_of_cat_info = {cat_name[i]: img_of_cat_cont[i] for i in all_cat_num}
    bbox_of_cat_info = {cat_name[i]:bbox_of_cat_cont[i] for i in all_cat_num}
    print('all img num:{},all bbox num:{},all category num:{}'.format(len(all_img_num), len(all_bbox_mum),
                                                                      len(all_cat_num)))
    print('cat_name:{}'.format(cat_name))
    print('bbox num info:{}'.format(bbox_of_cat_info))
    print('img num info:{}'.format(img_of_cat_info))

file = COCO('/home/wang/gitdemo/ifly_xdet_tools/val.json')
json_info_print(file)