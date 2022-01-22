import tensorflow as tf
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default='test_text_small.txt', help="path to text file to scan", type=str)
    parser.add_argument("--dvc", default='cpu', help="device (cpu/cuda)", type=str)
    args = parser.parse_args()
    
    file = args.file
    
    tf.keras.backend.clear_session()

    model_path = "tf_graph"
    
    model = tf.saved_model.load(model_path)
    recognize_func = model.signatures['serving_default']
    
    if args.dvc == 'cpu':
        with tf.device("/CPU:0"):
            detections = recognize_func(tf.constant(file)[tf.newaxis,...])
    else:
        detections = recognize_func(tf.constant(file)[tf.newaxis,...])

    NE = [detections['output_0'][0].numpy()[i].decode() for i in range(10)]
    
    for elm in NE:
        print(elm)