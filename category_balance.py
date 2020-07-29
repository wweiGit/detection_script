import json
import os
import cv2
import copy
from numpy import random
import math
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
img_dir = '/home/4T/algorithm/wangwei/dataset/ifly/train_set'
json_path = '/home/4T/algorithm/wangwei/dataset/ifly/annotations/train_add_one.json'
out_path = '/home/4T/algorithm/wangwei/dataset/ifly/new_img'
new_ann_id=100000

#sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def cate_img_num(cate_id,json_file):
    cate_img_id = set( ann['image_id']  for ann in json_file['annotations'] if ann['category_id']==cate_id )
    return len(cate_img_id)


def img_bbox_resize(new_img_id,img,anns):
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    interp = interp_method[random.randint(0, len(interp_method) - 1)]
    img_h,img_w,img_c = img.shape
    random_s = random.uniform(0.8,1.2)
    img = cv2.resize(img,(int(img_h*random_s),int(img_w*random_s)),interpolation=interp)
    new_img_name = str(new_img_id)+'.jpg'
    img_res = []
    img_res.append({'file_name':new_img_name,'height':int(img_h*random_s),'width':int(img_w*random_s),'id':new_img_name})
    ann_res = []
    print(anns)
    for ann in anns:
       # print('15')
        ann['area'] = int(img_h*random_s*img_w*random_s)
        ann['image_id'] = new_img_id
        ann['bbox'][0]*=random_s
        ann['bbox'][1]*=random_s
        ann['bbox'][2]*=random_s
        ann['bbox'][3]*=random_s
        ann_res.append(ann)
    return img,img_res,ann_res

def random_flip(img,img_res,ann_res):
    if random.random()>0.5:
        _,w_img,_ = img.shape
        img = img[:,::-1,:]
        for ann in ann_res:
            ann['bbox'][0] = w_img-ann['bbox'][2]
    else:
        h_img, _ ,_  = img.shape
        img = img[::-1, :, :]
        for ann in ann_res:
            ann['bbox'][1] = h_img-ann['bbox'][3]
    return img,img_res,ann_res

def rota_aug(img,anns):
    bbxs = [[ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]
             +ann['bbox'][3]] for ann in anns]
    seq = iaa.Sequential([
        #iaa.Flipud(0.5),  # vertically flip 20% of all images
        #iaa.Fliplr(0.5),  # 镜像
        #iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        #iaa.GaussianBlur(sigma=(0, 3.0)),
        #iaa.GaussianBlur(0.5),
        #iaa.Crop(percent=(0, 0.1)),
        iaa.Rot90((1,3),keep_size=False),
        iaa.Affine(
            scale = {"x":(0.8,1.2),"y":(0.8,1.2)},
            fit_output=True
        )
            #translate_px={"x": 15, "y": 15},
            #scale=(0.8, 1.2),
            #scale = {"x":(0.8,1.2),"y":(0.8,1.2)}
            #rotate=([0,90,180,270],fit_output=True)
      
          #translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    boxes_img_list = []
    for bbx in bbxs:
        boxes_img_list.append(BoundingBox(x1 = bbx[0],y1 = bbx[1],x2 = bbx[2],y2 = bbx[3]))
    bbs = BoundingBoxesOnImage(boxes_img_list,shape=img.shape)
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    return image_aug,bbs_aug


def img_ann_aug(new_img_id,img,anns,json_file):
    #img_h, img_w, img_c = img.shape
    image_aug,bbs_aug = rota_aug(img,anns)
    #for i in range(len(bbs_aug.bounding_boxes)):
        #after_aug.append(bbs_aug.bounding_boxes[i])
    img_aug_h,img_aug_w,img_c = image_aug.shape
    for i,ann in enumerate(anns):
        global new_ann_id
        new_ann_id+=1
        #print(new_ann_id)
        ann_tmp = copy.deepcopy(anns)
       # print(ann_tmp[0])
        #print(ann_tmp[0])
        ann_tmp[0]['image_id'] = new_img_id
        #如果坐标不完整则跳过
        '''
        if bbs_aug.bounding_boxes[i].x1<0 or bbs_aug.bounding_boxes[i].y1<0 \
                or bbs_aug.bounding_boxes[i].x2>img_w or bbs_aug.bounding_boxes[i].y2>img_h:
            print('aug bbox out of img! bbox:{}'.format([bbs_aug.bounding_boxes[i].x1,bbs_aug.bounding_boxes[i].y1,
                 bbs_aug.bounding_boxes[i].x2,bbs_aug.bounding_boxes[i].y2]))
            return json_file
        '''
        ann_tmp[0]['bbox'] = [int(max(bbs_aug.bounding_boxes[i].x1,0)),int(max(bbs_aug.bounding_boxes[i].y1,0)),
                       int(min(bbs_aug.bounding_boxes[i].x2-bbs_aug.bounding_boxes[i].x1,img_aug_w)),
                       int(min(bbs_aug.bounding_boxes[i].y2-bbs_aug.bounding_boxes[i].y1,img_aug_h))]
        ann_tmp[0]['area'] = int((bbs_aug.bounding_boxes[i].x2-bbs_aug.bounding_boxes[i].x1)*(bbs_aug.bounding_boxes[i].y2-bbs_aug.bounding_boxes[i].y1))
        ann_tmp[0]['id'] = new_ann_id
        json_file['annotations'].append(ann_tmp[0])
    new_img_name = str(new_img_id) + '.jpg'
    json_file['images'].append({'file_name': new_img_name, 'height': int(img_aug_h), 'width': int(img_aug_w),
                                'id': new_img_id})
    new_img_path = os.path.join(out_path, str(new_img_id) + '.jpg')
    cv2.imwrite(new_img_path, image_aug)
    return json_file

def cate_img_num_all_cont(json_file):
    cate_id_cont = set(cate['id'] for cate in json_file['categories'])
    #print(cate_id_cont)
    cate_img_num_all = dict()
    for cate_id in cate_id_cont:
        #此类别所拥有的图片数量
        cate_img_num_all[cate_id] = len(
            set(ann['image_id'] for ann in json_file['annotations'] if ann['category_id'] == cate_id))
    print("cate_img_num_info:{}".format(cate_img_num_all.values(), min(cate_img_num_all.values())))
    return cate_img_num_all

def cate_img_num(cate_id,json_file):
    return len(set(ann['image_id'] for ann in json_file['annotations'] if ann['category_id'] == cate_id))

def main():
    json_file = json.load(open(json_path))
    #print(len(json_file['images']))
    json_file_tmp = copy.deepcopy(json_file)
    new_img_id = 800000
    cate_img_num_all = cate_img_num_all_cont(json_file)
    print(cate_img_num_all)
    img_copy_times = dict()
    for cate_id in cate_img_num_all.keys():
        #此类别拥有的图片数量小于600时，计算每张图要copy的次数
        if cate_img_num_all[cate_id] < 600:
            img_copy_times[cate_id] = math.ceil((600 - cate_img_num_all[cate_id]) / cate_img_num_all[cate_id])
        else:
            img_copy_times[cate_id] = 0
    finish = 0
    while True:
        if finish==1:
            break
        for ann_tmp in json_file_tmp['annotations']:
            img_num = cate_img_num(ann_tmp['category_id'],json_file)
            if img_num>600:
               continue
            copy_time = img_copy_times[ann_tmp['category_id']]
            img_id_tmp = ann_tmp['image_id']
            #如果该图片中拥有某类别数量很多超过1000，则跳过该图片
            ann_deal_before = [ann for ann in json_file_tmp['annotations'] if ann['image_id'] == img_id_tmp]
            img_cate_id_cont = set(ann['category_id'] for ann in ann_deal_before)
            cate_img_nums = [cate_img_num(img_cate_id,json_file) for img_cate_id in img_cate_id_cont]
            if max(cate_img_nums)>1000:
                print('img cate have bigger 1000!')
                continue
            img_name = str(img_id_tmp)+'.jpg'
            img_path = os.path.join(img_dir,img_name)
            if os.path.exists(img_path):
                img_deal_before = cv2.imread(img_path,-1)
            else:
                print('img not exists:{}!'.format(img_path))
                continue
            for i in range(copy_time):
                #print('copy_time:{}'.format(copy_time))
                new_img_id +=1
                if (new_img_id-800000)%200==0:
                    print("new img num:{}".format(new_img_id-800000))
                json_file = img_ann_aug(new_img_id,img_deal_before,ann_deal_before,json_file)
            cate_img_num_all = cate_img_num_all_cont(json_file)
            if min(cate_img_num_all.values()) > 600:
                finish = 1
                break
    #print(len(json_file['images']))
    json.dump(json_file,open('/home/4T/algorithm/wangwei/dataset/ifly/train_cate_balance.json','w'))
    print('Finish category balance!')
   

if __name__=='__main__':
    main()
    



