from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import json
json_path = '/home/wang/data/ifly_xdet/xdet_data/json/train.json'


def json_info_print(file):
    all_img_num = file.getImgIds()
    all_bbox_mum = file.getAnnIds()
    all_cat_num = file.getCatIds()
    cat_name_info = file.loadCats(ids=all_cat_num)
    #print(cat_name_info)
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

file = COCO(json_path)
json_info_print(file)
#json_file = json.load(open(json_path))
#print(json_file.keys())
#anns = json_file['annotations']
#ann_cate =set(ann['category_id'] for ann in anns)
#print(ann_cate)
#cate = json_file['categories']
#print(cate[0])
