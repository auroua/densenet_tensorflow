from pycocotools.coco import COCO
import tensorflow as tf
import logging
import numpy as np


def gen_coco_obj(anno_filepath):
    return COCO(anno_filepath)


def gen_categories(coco_obj):
    cats = coco_obj.loadCats(coco_obj.getCatIds())
    cat2ind = {}
    ind2cat = {}
    for cat in cats:
        ind2cat[cat['id']] = cat['name']
        cat2ind[cat['name']] = cat['id']
    logging.info(cat2ind)
    logging.info(ind2cat)
    return cat2ind, ind2cat


def getImgs(coco_obj, categories):
    catIds = coco_obj.getCatIds(catNms=categories)
    imgIds = coco_obj.getImgIds(catIds=catIds)
    logging.info(len(imgIds))
    return imgIds


def get_annots(coco_obj, img, cats2index):
    annIds = coco_obj.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco_obj.loadAnns(annIds)
    bboxs = [(ann['category_id'], ann['bbox']) for ann in anns]
    return bboxs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    anno_filepath = '/home/aurora/workspaces/data/COCO/coco/annotations/instances_train2014.json'
    coco = gen_coco_obj(anno_filepath)
    cat2ind, ind2cat = gen_categories(coco)
    imgIds = getImgs(coco, cat2ind.keys())
    imgs = coco.loadImgs(imgIds)
    for img in imgs:
        bboxs = get_annots(coco, img, ind2cat)
        print(bboxs)