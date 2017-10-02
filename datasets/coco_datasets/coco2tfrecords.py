import tensorflow as tf
import cv2
from coco_utils import gen_coco_obj, gen_categories, getImgs, get_annots
import logging
import os
import io
import hashlib
import utils.dataset_util as dataset_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', '/home/aurora/workspaces/data/COCO/coco/images/train2014', 'coco image file dir')
flags.DEFINE_string('ann_dir', '/home/aurora/workspaces/data/COCO/coco/annotations/instances_train2014.json', 'annotation file dir')
flags.DEFINE_string('image_val_dir', '/home/aurora/workspaces/data/COCO/coco/images/val2014', 'coco image file dir')
flags.DEFINE_string('ann_val_dir', '/home/aurora/workspaces/data/COCO/coco/annotations/instances_val2014.json', 'annotation file dir')
flags.DEFINE_string('image_test_dir', '/home/aurora/workspaces/data/COCO/coco/images/test2014', 'coco image file dir')
flags.DEFINE_string('ann_test_dir', '/home/aurora/workspaces/data/COCO/coco/annotations/image_info_test2014.json', 'annotation file dir')
flags.DEFINE_string('output_dir', '/home/aurora/workspaces/data/COCO/tfrecords', 'tfrecords output dir')
flags.DEFINE_string('type', 'test', 'tf records category')

FLAGS = flags.FLAGS

if __name__ == '__main__':
    '''
        modify three locations, to generate different type tfrecords:
        1. line 18 tfrecord type  reference line 29
        2. line 33 coco tfrecords annotation file dir
        3. line 40 coco image file dir
    '''
    logging.basicConfig(level=logging.WARN)

    tfrecords_files = os.path.join(FLAGS.output_dir, FLAGS.type+'.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecords_files)

    coco = gen_coco_obj(FLAGS.ann_test_dir)
    cat2ind, ind2cat = gen_categories(coco)
    imgIds = getImgs(coco, cat2ind.keys())
    imgs = coco.loadImgs(imgIds)
    for img_info in imgs:
        logging.info(img_info)
        img_path = os.path.join(FLAGS.image_test_dir, img_info['file_name'])
        # img = cv2.imread(img_path)
        # cv2.imshow('img_name', img)
        # cv2.waitKey(0)
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)

        key = hashlib.sha256(encoded_jpg).hexdigest()

        height = img_info['height']
        width = img_info['width']

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []

        truncated = []
        poses = []
        difficult_obj = []

        bboxs = get_annots(coco, img_info, ind2cat)
        for bbox in bboxs:
            box = bbox[1]
            xmin.append(float(box[0]) / width)
            ymin.append(float(box[1]) / height)
            xmax.append((float(box[0]) + float(box[2])) / width)
            ymax.append((float(box[1]) + float(box[3])) / height)

            classes.append(bbox[0])
            classes_text.append(ind2cat[bbox[0]].encode('utf8'))

            truncated.append(0)
            poses.append('Frontal'.encode('utf8'))
            difficult_obj.append(0)

        example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    img_info['file_name'].encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                    img_info['file_name'].encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
                'image/object/truncated': dataset_util.int64_list_feature(truncated),
                'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
        writer.write(example.SerializeToString())
    writer.close()