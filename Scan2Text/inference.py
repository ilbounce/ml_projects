import tensorflow as tf
import tensorflow.keras as k
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import codecs

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default='doc.jpg', help="path to image to scan", type=str)
    parser.add_argument("--dvc", default='cpu', help="device (cpu/cuda)", type=str)
    args = parser.parse_args()
    
    img = args.img
    
    k.backend.clear_session()

    model_path = "ScanToText"
    
    if args.dvc == 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "12"
        tf.get_logger().setLevel('INFO')
        with tf.device('/cpu:0'):
            model = tf.saved_model.load(model_path)
            recognize_func = model.signatures['serving_default']
            
    else:
        model = tf.saved_model.load(model_path)
        recognize_func = model.signatures['serving_default']
    
    encoded = tf.io.read_file(img)
    detections = recognize_func(encoded[tf.newaxis,...])
    
    print(detections['output_0'].numpy()[0].decode('utf-8').split('\n'))
    
    with codecs.open(img[:-4] + '_text.txt', 'w', 'utf-8') as f:
        for item in detections['output_0'].numpy()[0].decode('utf-8').split('\n'):
            f.write("%s\n" % item)
    
    image = plt.imread(img)
    fig = plt.figure(figsize=(20,20))
    ax = plt.axes()
    ax.imshow(image)
    edgecolor = 'r'
    for word in detections['output_1'].numpy()[0]:
        edgecolor = 'r' if edgecolor == 'b' else 'b'
        if (word == [0,0,0,0]).all():
            continue
        else:
            y_min = word[0]
            x_min = word[1]
            w = word[3] - word[1]
            h = word[2] - word[0]
            if w > 0 and h > 0:
                rect = patches.Rectangle((x_min+1, y_min), w-1, h, linewidth=1, edgecolor=edgecolor, facecolor='none')
                ax.add_patch(rect)
    plt.savefig(img[:-4] + '_annotated.jpg')