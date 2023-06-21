import pickle, os, time
from keras.layers import Input
from keras.models import Model

import config
import faster_rcnn.vgg as nn
from faster_rcnn.utils import *
from faster_rcnn.roi_helpers import *

class FasterRCNNDetector(object):

    def __init__(self, model_path):
        self.model_path = model_path

        if os.path.exists('config.pickle'):
            with open('config.pickle', 'rb') as f:
                self.cfg = pickle.load(f)
        else:
            self.cfg = config.Config()
            print('Not found previous train and saved config.pickle file. may lose class map info.')
        self._init_model()

    def _init_model(self):
        self.cfg.use_horizontal_flips = False
        self.cfg.use_vertical_flips = False
        self.cfg.rot_90 = False

        if self.cfg.class_mapping == None:
            self.cfg.class_mapping = {'articulated_truck': 0,
                                        'bicycle': 1,
                                        'bus': 2,
                                        'car': 3,
                                        'motorcycle': 4,
                                        'motorized_vehicle': 5,
                                        'non-motorized_vehicle': 6,
                                        'pedestrian': 7,
                                        'pickup_truck': 8,
                                        'single_unit_truck': 9,
                                        'work_van': 10}

        class_mapping = self.cfg.class_mapping
        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        self.class_mapping = {v: k for k, v in class_mapping.items()}
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, 512)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.cfg.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        shared_layers = nn.nn_base(img_input, trainable=False)

        # define the RPN, built on the base layers
        num_anchors = len(self.cfg.anchor_box_scales) * len(self.cfg.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        classifier = nn.classifier(feature_map_input, roi_input, self.cfg.num_rois, nb_classes=len(class_mapping),
                                   trainable=True)

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)

        self.model_classifier = Model([feature_map_input, roi_input], classifier)

        if os.path.exists(self.model_path):
            model_path = self.model_path
        else:
            model_path = self.cfg.model_path
        print('Loading weights from {}'.format(model_path))

        self.model_rpn.load_weights(model_path, by_name=True)
        self.model_classifier.load_weights(model_path, by_name=True)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def detect_on_image(self, img):
        tic = time.time()

        X, ratio = format_img(img, self.cfg)
        X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)

        # this is result contains all boxes, which is [x1, y1, x2, y2]
        result = rpn_to_roi(Y1, Y2, self.cfg, overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        result[:, 2] -= result[:, 0]
        result[:, 3] -= result[:, 1]
        bbox_threshold = 0.8

        # apply the spatial pyramid pooling to the proposed regions
        boxes = dict()
        for jk in range(result.shape[0] // self.cfg.num_rois + 1):
            rois = np.expand_dims(result[self.cfg.num_rois * jk:self.cfg.num_rois * (jk + 1), :], axis=0)
            if rois.shape[1] == 0:
                break
            if jk == result.shape[0] // self.cfg.num_rois:
                # pad R
                curr_shape = rois.shape
                target_shape = (curr_shape[0], self.cfg.num_rois, curr_shape[2])
                rois_padded = np.zeros(target_shape).astype(rois.dtype)
                rois_padded[:, :curr_shape[1], :] = rois
                rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
                rois = rois_padded

            [p_cls, p_regr] = self.model_classifier_only.predict([F, rois])

            for ii in range(p_cls.shape[1]):
                if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                    continue

                cls_num = np.argmax(p_cls[0, ii, :])
                if cls_num not in boxes.keys():
                    boxes[cls_num] = []
                (x, y, w, h) = rois[0, ii, :]
                try:
                    (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.cfg.classifier_regr_std[0]
                    ty /= self.cfg.classifier_regr_std[1]
                    tw /= self.cfg.classifier_regr_std[2]
                    th /= self.cfg.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except Exception as e:
                    print(e)
                    pass
                boxes[cls_num].append(
                    [self.cfg.rpn_stride * x, self.cfg.rpn_stride * y, self.cfg.rpn_stride * (x + w), self.cfg.rpn_stride * (y + h),
                     np.max(p_cls[0, ii, :])])
        # add some nms to reduce many boxes
        for cls_num, box in boxes.items():
            boxes_nms = non_max_suppression_fast(box, overlap_thresh=0.5, max_boxes = 300)
            boxes[cls_num] = boxes_nms
            print(self.class_mapping[cls_num] + ":")
            for b in boxes_nms:
                b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
                print('{} prob: {}'.format(b[0: 4], b[-1]))
        img = draw_boxes_and_label_on_image_cv2(img, self.class_mapping, boxes)
        print('Elapsed time = {}'.format(time.time() - tic))
        cv2.imshow('image', img)

        result_path = './test on imgs/output_img/{}.png'.format('result')
        print('result saved into ', result_path)
        cv2.imwrite(result_path, img)
        cv2.waitKey(0)

    def detect_on_video(self, v):
        pass