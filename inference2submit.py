import json
import argparse
import os
import numpy as np
import copy
parser = argparse.ArgumentParser(description='inference2submit')
parser.add_argument('dt',help='inference json file path')
parser.add_argument('--out',help='inference json file path')
args = parser.parse_args()
test_img_path = '/home/wang/data/ifly_xdet/test1'
img_name_list = os.listdir(test_img_path)
img_name_list.sort()
#data_cont = json.load(open(args.dt))
#cate_cont = set(cate['category_id'] for cate in data_cont)
#print(cate_cont)\
submit_cate = {'knife': 0, 'scissors': 1, 'lighter': 2, 'zippooil': 3, 'pressure': 4, 'slingshot': 5,
               'handcuffs': 6, 'nailpolish': 7,'powerbank': 8, 'firecrackers': 9}

def main():
    data_cont = json.load(open(args.dt))
    print(len(data_cont))
    submit=[]
    for i,img_name in enumerate(img_name_list):

        bboxs = [bbox for bbox in data_cont if bbox['file_name'] == img_name]
        bbox_parser = [[] for j in range(10)]
        print(len(bboxs))
        for bbox in bboxs:

            bbox_score = [bbox['bbox'][0],bbox['bbox'][1],bbox['bbox'][0]+bbox['bbox'][2],bbox['bbox'][1]+bbox['bbox'][3]]
            bbox_score.append(bbox['score'])
            submit_cate[bbox['category_name']]
            bbox_parser[submit_cate[bbox['category_name']]-1].append(bbox_score)
        #print(bbox_parser)

        for bbox_ in bbox_parser:
            if len(bbox_)>1:
                bbox_tmp =copy.deepcopy(bbox_)
                bbox_tmp = np.array(bbox_tmp)
                bbox_ = bbox_tmp[np.lexsort(-bbox_tmp.T)]
                #print('a bbox_:{}'.format(bbox_))

        submit.append(bbox_parser)
        print('finish img num:{}'.format(i))
    json.dump(submit,open(args.out,'w'))
    #print(submit[:10])

if __name__ == '__main__':
    main()