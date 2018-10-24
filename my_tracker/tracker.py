import os
import time

import cv2
import scipy.io
import tensorflow.contrib.lite as tflite

from main import store_dir
from my_tracker.convolutional import *
from src.crops import *
from src.parse_arguments import parse_arguments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Tracker:
    def __init__(self):
        hp, evaluation, run, env, design = parse_arguments()
        final_score_sz = hp.response_up * (design.score_sz - 1) + 1

        self.image_input = tf.placeholder(tf.float32, name='img_in', shape=(360, 640, 3))

        self.pos_x_ph = tf.placeholder(tf.float64, name='pos_x_ph', shape=(1,))
        self.pos_y_ph = tf.placeholder(tf.float64, name='pos_y_ph', shape=(1,))
        # target的尺寸 size
        self.z_sz_ph = tf.placeholder(tf.float64, name='z_sz_ph', shape=(1,))
        # 对search input 进行三种系数的缩放后的输入结果
        #   将search input进行不同大小的缩放，满足当target的scale出现变化时，
        #   tracker也能保证sampler和search input中的target大小尽可能相似
        self.x_sz0_ph = tf.placeholder(tf.float64, name='x_sz0_ph', shape=(1,))
        self.x_sz1_ph = tf.placeholder(tf.float64, name='x_sz1_ph', shape=(1,))
        self.x_sz2_ph = tf.placeholder(tf.float64, name='x_sz2_ph', shape=(1,))

        # self.pos_x_ph = tf.placeholder(tf.float64, name='pos_x_ph', )
        # self.pos_y_ph = tf.placeholder(tf.float64, name='pos_y_ph', )
        # self.z_sz_ph = tf.placeholder(tf.float64, name='z_sz_ph', )
        # self.x_sz0_ph = tf.placeholder(tf.float64, name='x_sz0_ph', )
        # self.x_sz1_ph = tf.placeholder(tf.float64, name='x_sz1_ph', )
        # self.x_sz2_ph = tf.placeholder(tf.float64, name='x_sz2_ph', )
        
        self.template_x, self.templates_z, self.scores, \
        self.crop_x, self.crop_z, \
        self.padded_x, self.padded_z = _build_tracking_graph(self.image_input, final_score_sz, design, env,
                                                             self.pos_x_ph, self.pos_y_ph, self.z_sz_ph,
                                                             self.x_sz0_ph, self.x_sz1_ph, self.x_sz2_ph)
        
        self.scale_factors = hp.scale_step ** np.linspace(-np.ceil(hp.scale_num / 2), np.ceil(hp.scale_num / 2),
                                                          hp.scale_num)
        self.scale_factors = np.expand_dims(self.scale_factors, axis=-1)
        self.final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        
        self.template_data = None
        """
        region 形式：
            中心点坐标 + 目标的长宽
        """
        self.last_pos_x = None
        self.last_pos_y = None
        self.target_w = 160.
        self.target_h = 160.
        
        # cosine window to penalize large displacements
        self.hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
        penalty = np.transpose(self.hann_1d) * self.hann_1d
        self.penalty = penalty / np.sum(penalty)
        
        self.context = design.context * (self.target_w + self.target_h)
        self.z_sz = np.sqrt(np.prod((self.target_w + self.context) * (self.target_h + self.context)))
        self.x_sz = float(design.search_sz) / design.exemplar_sz * self.z_sz
        
        self.hp = hp
        self.design = design
    
    def stop_tracking(self):
        self.template_data = None
    
    def tracking(self, frames):
        bbox_res = []
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            first_frame = frames[0]

            """
            对于坐标的一些设定：
                设一个图片的横向为X轴，纵向为Y轴，在opencv取出的img的matrix中，
                Y轴对应的矩阵的0 dim，X轴对应的是矩阵的1dim
                转换时需要自己进行一下反转
                随后使用opencv的方法对图片进行修改时，又恢复的之前横向X纵向Y的顺序
            """
            print("first_frame.shape %s" % str(first_frame.shape))
            self.last_pos_x = np.array([first_frame.shape[1] / 2])
            self.last_pos_y = np.array([first_frame.shape[0] / 2])
            """
            bbox:
                左上点和右下点坐标
            """
            bbox_res.append(((int(self.last_pos_x - self.target_w / 2),
                              int(self.last_pos_y - self.target_h / 2)),
                             (int(self.last_pos_x + self.target_w / 2),
                              int(self.last_pos_y + self.target_h / 2))))
            # file_writer = tf.summary.FileWriter('logs', sess.graph)
            # print("tensorboard log created")

            image_, templates_z_ = sess.run(fetches=[self.image_input, self.templates_z, ],
                                                      feed_dict={
                                                          self.pos_x_ph: self.last_pos_x,
                                                          self.pos_y_ph: self.last_pos_y,
                                                          self.z_sz_ph: np.array([160], dtype=np.float64),
                                                          self.image_input: first_frame
                                                      })
            print("first frame template encoded")
            self.template_data = templates_z_
            
            for each_frame in range(1, len(frames)):
                print("processed frame %d" % each_frame)
                start_time = time.time()
                each_frame = frames[each_frame]
                scaled_exemplar = self.z_sz * self.scale_factors
                scaled_search_area = self.x_sz * self.scale_factors
                scaled_target_w = self.target_w * self.scale_factors
                scaled_target_h = self.target_h * self.scale_factors
                image_, scores_, z_, x_ = sess.run(
                    [self.image_input, self.scores, self.templates_z, self.template_x],
                    feed_dict={
                        self.image_input: each_frame,
                        self.pos_x_ph: np.array(self.last_pos_x),
                        self.pos_y_ph: np.array(self.last_pos_y),
                        self.x_sz0_ph: scaled_search_area[0],
                        self.x_sz1_ph: scaled_search_area[1],
                        self.x_sz2_ph: scaled_search_area[2],
                        self.templates_z: np.squeeze(self.template_data),
                    })
                scores_ = np.squeeze(scores_)
                # print("score %s\nx_ %s\nz_ %s" % (str(scores_), str(x_), str(z_)))
                # penalize change of scale
                scores_[0, :, :] = self.hp.scale_penalty * scores_[0, :, :]
                scores_[2, :, :] = self.hp.scale_penalty * scores_[2, :, :]
                # find scale with highest peak (after penalty)
                new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
                # update scaled sizes
                self.x_sz = (1 - self.hp.scale_lr) * self.x_sz + self.hp.scale_lr * scaled_search_area[new_scale_id]
                target_w = (1 - self.hp.scale_lr) * self.target_w + self.hp.scale_lr * scaled_target_w[new_scale_id]
                target_h = (1 - self.hp.scale_lr) * self.target_h + self.hp.scale_lr * scaled_target_h[new_scale_id]
                # select response with new_scale_id
                score_ = scores_[new_scale_id, :, :]
                score_ = score_ - np.min(score_)
                score_ = score_ / np.sum(score_)
                # apply displacement penalty
                score_ = (1 - self.hp.window_influence) * score_ + self.hp.window_influence * self.penalty
                self.last_pos_x, self.last_pos_y = _update_target_position(self.last_pos_x, self.last_pos_y, score_,
                                                                           self.final_score_sz, self.design.tot_stride,
                                                                           self.design.search_sz, self.hp.response_up,
                                                                           self.x_sz)
                self.last_pos_x = np.array(self.last_pos_x)
                self.last_pos_y = np.array(self.last_pos_y)
                # convert <cx,cy,w,h> to <x1,y1,x2,y2> and save output
                bbox_res.append((
                    (int(self.last_pos_x - target_w / 2),
                     int(self.last_pos_y - target_h / 2)),
                    (int(self.last_pos_x + target_w / 2),
                     int(self.last_pos_y + target_h / 2))
                ))

                # update the target representation with a rolling average
                if self.hp.z_lr > 0:
                    if len(self.z_sz.shape) == 0:
                        self.z_sz = np.array([self.z_sz])
                        print(self.z_sz.shape)
                    new_templates_z_ = sess.run([self.templates_z],
                                                feed_dict={
                                                    self.pos_x_ph: np.array(self.last_pos_x),
                                                    self.pos_y_ph: np.array(self.last_pos_y),
                                                    self.z_sz_ph: np.array(self.z_sz),
                                                    self.image_input: image_
                                                })
                    
                    self.template_data = (1 - self.hp.z_lr) * np.asarray(
                        self.template_data) + self.hp.z_lr * np.asarray(new_templates_z_)
                # update template patch size
                self.z_sz = (1 - self.hp.scale_lr) * self.z_sz + self.hp.scale_lr * scaled_exemplar[new_scale_id]
    
                print('bbox %s' % str(bbox_res[-1]))
                print("cost time %f" % (time.time() - start_time))
                frame = each_frame
                frame = cv2.rectangle(frame, bbox_res[-1][0], bbox_res[-1][1], (0, 0, 255))
                cv2.imshow('tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        return bbox_res

    def save_tflite(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print(store_dir)
            converter = tflite.TocoConverter.from_session(sess,
                                                          [self.image_input,
                                                           self.pos_x_ph, self.pos_y_ph,
                                                           self.x_sz0_ph, self.x_sz1_ph,
                                                           self.x_sz2_ph, self.templates_z],
        
                                                          [self.image_input, self.scores,
                                                           self.templates_z, self.template_x])
            tflite_model = converter.convert()
            open(os.path.join(store_dir, "converted_model.tflite", "wb")).write(tflite_model)
            print(".tflite saved")
            


def _build_tracking_graph(image, final_score_sz, design, env,
                          pos_x_ph, pos_y_ph, z_sz_ph,
                          x_sz0_ph, x_sz1_ph, x_sz2_ph, ):
    image = tf.convert_to_tensor(image)
    frame_sz = tf.shape(image)  # 获取图片的channel和尺寸
    # used to pad the crops
    if design.pad_with_image_mean:
        avg_chan = tf.reduce_mean(image, axis=(0, 1), name='avg_chan')
    else:
        avg_chan = None
    # pad with if necessary
    #   当输入的大小可能发生变换时，为了保证神经网络的输出size保持不变，需要对输入进行padding
    frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
    frame_padded_z = tf.cast(frame_padded_z, tf.float32)
    # extract tensor of z_crops
    # 根据的上一帧target的位置，将其crop出来，作为sampler为search过程提供比对目标
    z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, design.exemplar_sz)
    
    # 处理search input x
    #     首先将其按不同系数scale后的结果padding到最大scale input的大小
    frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan)
    frame_padded_x = tf.cast(frame_padded_x, tf.float32)
    # extract tensor of x_crops (3 scales)
    # 根据target可能的位置和search的区域，将三种scale后的input data进行crop，做输入神经网络前数据
    # size的预处理
    x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph,
                              design.search_sz)
    # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
    net_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), env.root_pretrained, design.net)
    template_z, templates_x, p_names_list, p_val_list = _create_siamese(net_path,
                                                                        x_crops, z_crops)
    template_z = tf.squeeze(template_z)
    templates_z = tf.stack([template_z, template_z, template_z])
    # compare templates via cross-correlation
    scores = _match_templates(templates_z, templates_x, p_names_list, p_val_list)
    # upsample the score maps
    scores_up = tf.image.resize_images(scores, [final_score_sz, final_score_sz],
                                       method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
    return templates_x, templates_z, scores_up, x_crops, z_crops, frame_padded_x, frame_padded_z


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


# the follow parameters *have to* reflect the design of the network to be imported
# 一些对网络参数的设置
_conv_stride = np.array([2, 1, 1, 1, 1])
_filtergroup_yn = np.array([0, 1, 0, 1, 1], dtype=bool)
_bnorm_yn = np.array([1, 1, 1, 1, 0], dtype=bool)
_relu_yn = np.array([1, 1, 1, 1, 0], dtype=bool)
_pool_stride = np.array([2, 1, 0, 0, 0])  # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), (
    'These arrays of flags should have same length')
assert all(_conv_stride) >= True, 'The number of conv layers is assumed to define the depth of the network'
_num_layers = len(_conv_stride)


# import pretrained Siamese network from matconvnet
def _create_siamese(net_path, net_x, net_z):
    # read mat file from net_path and start TF Siamese graph from placeholders X and Z
    params_names_list, params_values_list = _import_from_matconvnet(net_path)
    
    # loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
    for i in range(_num_layers):
        print('> Layer ' + str(i + 1))
        # conv
        conv_W_name = _find_params('conv' + str(i + 1) + 'f', params_names_list)[0]
        conv_b_name = _find_params('conv' + str(i + 1) + 'b', params_names_list)[0]
        print('\t\tCONV: setting ' + conv_W_name + ' ' + conv_b_name)
        print('\t\tCONV: stride ' + str(_conv_stride[i]) + ', filter-group ' + str(_filtergroup_yn[i]))
        conv_W = params_values_list[params_names_list.index(conv_W_name)]
        conv_b = params_values_list[params_names_list.index(conv_b_name)]
        # batchnorm
        if _bnorm_yn[i]:
            bn_beta_name = _find_params('bn' + str(i + 1) + 'b', params_names_list)[0]
            bn_gamma_name = _find_params('bn' + str(i + 1) + 'm', params_names_list)[0]
            bn_moments_name = _find_params('bn' + str(i + 1) + 'x', params_names_list)[0]
            print('\t\tBNORM: setting ' + bn_beta_name + ' ' + bn_gamma_name + ' ' + bn_moments_name)
            bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
            bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
            bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
            bn_moving_mean = bn_moments[:, 0]
            bn_moving_variance = bn_moments[:, 1] ** 2  # saved as std in matconvnet
        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
        
        # set up conv "block" with bnorm and activation
        net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b, 0, 1), _conv_stride[i],
                                  bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,
                                  filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i],
                                  scope='conv' + str(i + 1), reuse=False)
        
        # notice reuse=True for Siamese parameters sharing
        net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b, 0, 1), _conv_stride[i],
                                  bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance,
                                  filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i],
                                  scope='conv' + str(i + 1), reuse=True)
        
        # add max pool if required
        if _pool_stride[i] > 0:
            print('\t\tMAX-POOL: size ' + str(_pool_sz) + ' and stride ' + str(_pool_stride[i]))
            net_x = tf.nn.max_pool(net_x, [1, _pool_sz, _pool_sz, 1], strides=[1, _pool_stride[i], _pool_stride[i], 1],
                                   padding='VALID', name='pool' + str(i + 1))
            net_z = tf.nn.max_pool(net_z, [1, _pool_sz, _pool_sz, 1], strides=[1, _pool_stride[i], _pool_stride[i], 1],
                                   padding='VALID', name='pool' + str(i + 1))
    
    print()
    
    return net_z, net_x, params_names_list, params_values_list


def _import_from_matconvnet(net_path):
    print(net_path)
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    # organize parameters to import
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]
    return params_names_list, params_values_list


# find all parameters matching the codename (there should be only one)
def _find_params(x, params):
    matching = [s for s in params if x in s]
    assert len(matching) == 1, 'Ambiguous param name found'
    return matching


def _match_templates(net_z, net_x, params_names_list, params_values_list):
    # finalize network
    # z, x are [B, H, W, C]
    net_z = tf.transpose(net_z, perm=[1, 2, 0, 3])
    net_x = tf.transpose(net_x, perm=[1, 2, 0, 3])
    # z, x are [H, W, B, C]
    Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
    Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
    # assert B==Bx, ('Z and X should have same Batch size')
    # assert C==Cx, ('Z and X should have same Channels number')
    net_z = tf.reshape(net_z, (Hz, Wz, B * C, 1))
    net_x = tf.reshape(net_x, (1, Hx, Wx, B * C))
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1, 1, 1, 1], padding='VALID')
    # final is [1, Hf, Wf, BC]
    net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)
    # final is [B, Hf, Wf, C]
    net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
    # final is [B, Hf, Wf, 1]
    if _bnorm_adjust:
        bn_beta = params_values_list[params_names_list.index('fin_adjust_bnb')]
        bn_gamma = params_values_list[params_names_list.index('fin_adjust_bnm')]
        bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
        bn_moving_mean = bn_moments[:, 0]
        bn_moving_variance = bn_moments[:, 1] ** 2
        net_final = tf.layers.batch_normalization(net_final, beta_initializer=tf.constant_initializer(bn_beta),
                                                  gamma_initializer=tf.constant_initializer(bn_gamma),
                                                  moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
                                                  moving_variance_initializer=tf.constant_initializer(
                                                      bn_moving_variance),
                                                  training=False, trainable=False)
    
    return net_final
