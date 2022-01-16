import tensorflow as tf
import tensorflow.keras as k
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import collections

class Image2Text(tf.Module):

    def __init__(self, lines_detection_model_path, words_detection_model_path, words_ocr_model_path):
        super(Image2Text, self).__init__()

        self.lines_detection_model = tf.saved_model.load(lines_detection_model_path)
        self.lines_detection_threshold = 0.3

        self.words_detection_model = tf.saved_model.load(words_detection_model_path)
        self.words_detection_threshold = 0.3

        self.words_ocr_model = tf.saved_model.load(words_ocr_model_path)
        self.ocr_word_height = 25
        self.ocr_word_width = 300
        self.max_image_size = 1500
    
        self.OUTPUT = collections.namedtuple('output', 'text, boxes')

    @staticmethod
    def get_coordinates(boxes, scores, image_shape, threshold):
        scores = (scores > threshold)
        boxes = tf.boolean_mask(boxes, scores)

        width = image_shape[0]
        height = image_shape[1]
        absolutes = tf.cast(tf.stack([width, height, width, height], 0), tf.float32)
        boxes = tf.multiply(boxes, absolutes)

        return boxes

    @staticmethod
    def sort(tensor, col):
        return tf.gather(
            tensor,
            tf.nn.top_k(-tensor[:, col], k=tf.shape(tensor)[0]).indices
        )
    @staticmethod
    def intersect(tensor1, tensor2, projection):
        def f1():
            return tf.math.reduce_max([tensor1[0], tensor2[0]]), tf.math.reduce_min([tensor1[2], tensor2[2]])
        def f2():
            return tf.math.reduce_max([tensor1[1], tensor2[1]]), tf.math.reduce_min([tensor1[3], tensor2[3]])
        a, b = tf.cond(tf.constant(projection) == 'y',
               f1,
               f2)
        return b - a > 0
    
    def smart_sortX(self, tensor):
        sort_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        cur_group = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        begin = tf.constant(True)
        inter_detection = tf.constant(False)
        c = tf.constant(0)
        j = tf.constant(0)
        i = tf.constant(0)
        
        ALL_VARS = collections.namedtuple('ALL_VARS', 'c, j, begin, inter_detection, sort_tensor, cur_group')
        
        def loop_body(i, tensor, all_vars):
            
            def begin_condition(i, tensor, v):
                cur_group = v.cur_group.write(v.j, tensor[i])
                begin = tf.constant(False)
                
                return ALL_VARS(v.c, v.j, begin, v.inter_detection, v.sort_tensor, cur_group)
                
            def other_condition(i, tensor, v):
                jj = tf.constant(0)
                
                def intersect_cond(v):
                    inter_detection = tf.constant(True)
                    return ALL_VARS(v.c, v.j, v.begin, inter_detection, v.sort_tensor, v.cur_group)
                
                def not_intersect_cond(v):
                    inter_detection = v.inter_detection
                    return ALL_VARS(v.c, v.j, v.begin, inter_detection, v.sort_tensor, v.cur_group)
                
                _, _, _, v = tf.while_loop(lambda i, jj, tensor, v: tf.less(jj, v.j + 1),
                             lambda i, jj, tensor, v: (i,
                                                                  tf.add(jj, 1),
                                                                  tensor,
                                                                  tf.cond(self.intersect(tensor[i], v.cur_group.read(jj), 'x'),
                                                                         lambda: intersect_cond(v),
                                                                         lambda: not_intersect_cond(v)),),
                             [i, jj, tensor, v])
                
                def f1(v):
                    j = tf.add(v.j,1)
                    cur_group = v.cur_group.write(j, tensor[i])
                    return ALL_VARS(v.c, j, v.begin, v.inter_detection, v.sort_tensor, cur_group)
                    
                def f2(v):
                    cur_group_stack = self.sort(v.cur_group.stack(), 0)
                    jj = tf.constant(0)
                    
                    def body(jj, cur_group_stack, v):
                        sort_tensor = v.sort_tensor.write(v.c, cur_group_stack[jj])
                        c = tf.add(v.c, 1)
                        return ALL_VARS(c, v.j, v.begin, v.inter_detection, sort_tensor, v.cur_group)
                        
                    _, _, v = tf.while_loop(lambda jj, cur_group_stack, v: tf.less(jj, v.j+1),
                                 lambda jj, cur_group_stack, v: (tf.add(jj,1), cur_group_stack, body(jj, cur_group_stack, v),),
                                 [jj, cur_group_stack, v])
                    cur_group = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                    j = tf.constant(0)
                    cur_group = cur_group.write(j, tensor[i])
                    begin = tf.constant(False)
                    
                    return ALL_VARS(v.c, j, begin, v.inter_detection, v.sort_tensor, cur_group)
                
                v = tf.cond(v.inter_detection, 
                       lambda: f1(v),
                       lambda: f2(v))
                inter_detection = tf.constant(False)
                
                return ALL_VARS(v.c, v.j, v.begin, inter_detection, v.sort_tensor, v.cur_group)
                
                
            all_vars = tf.cond(all_vars.begin, 
                    lambda: begin_condition(i, tensor, all_vars), 
                    lambda: other_condition(i, tensor, all_vars))
            
            return all_vars
                
        _, _, all_vars = tf.while_loop(lambda i, tensor, v: tf.less(i, tf.shape(tensor)[0]), 
                    lambda i, tensor, v: (tf.add(i,1), tensor, loop_body(i,tensor,v),), 
                    [i, tensor, ALL_VARS(c, j, begin, inter_detection, sort_tensor, cur_group)])
        
        def last_group(v):
            cur_group_stack = self.sort(v.cur_group.stack(), 0)
            jj = tf.constant(0)
            
            def body(jj, cur_group_stack, v):
                sort_tensor = v.sort_tensor.write(v.c, cur_group_stack[jj])
                c = tf.add(v.c, 1)
                
                return ALL_VARS(c, v.j, v.begin, v.inter_detection, sort_tensor, v.cur_group)
            _, _, v = tf.while_loop(lambda jj, cur_group_stack, v: tf.less(jj, v.j+1),
                                          lambda jj, cur_group_stack, v: (tf.add(jj,1), cur_group_stack, body(jj, cur_group_stack, v),),
                                          [jj, cur_group_stack, v])
            return v
        
        def no_last_group(v):
            return v
        
        all_vars = tf.cond(all_vars.cur_group.size() != 0, lambda: last_group(all_vars), lambda: no_last_group(all_vars))
        
        return all_vars.sort_tensor.stack()

    def smart_sortY(self, tensor):
        sort_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        cur_group = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        begin = tf.constant(True)
        inter_detection = tf.constant(False)
        c = tf.constant(0)
        j = tf.constant(0)
        i = tf.constant(0)
        
        ALL_VARS = collections.namedtuple('ALL_VARS', 'c, j, begin, inter_detection, sort_tensor, cur_group')
        
        def loop_body(i, tensor, all_vars):
            
            def begin_condition(i, tensor, v):
                cur_group = v.cur_group.write(v.j, tensor[i])
                begin = tf.constant(False)
                
                return ALL_VARS(v.c, v.j, begin, v.inter_detection, v.sort_tensor, cur_group)
                
            def other_condition(i, tensor, v):
                jj = tf.constant(0)
                
                def intersect_cond(v):
                    inter_detection = tf.constant(True)
                    return ALL_VARS(v.c, v.j, v.begin, inter_detection, v.sort_tensor, v.cur_group)
                
                def not_intersect_cond(v):
                    inter_detection = v.inter_detection
                    return ALL_VARS(v.c, v.j, v.begin, inter_detection, v.sort_tensor, v.cur_group)
                
                _, _, _, v = tf.while_loop(lambda i, jj, tensor, v: tf.less(jj, v.j + 1),
                             lambda i, jj, tensor, v: (i,
                                                                  tf.add(jj, 1),
                                                                  tensor,
                                                                  tf.cond(self.intersect(tensor[i], v.cur_group.read(jj), 'y'),
                                                                         lambda: intersect_cond(v),
                                                                         lambda: not_intersect_cond(v)),),
                             [i, jj, tensor, v])
                
                def f1(v):
                    j = tf.add(v.j,1)
                    cur_group = v.cur_group.write(j, tensor[i])
                    return ALL_VARS(v.c, j, v.begin, v.inter_detection, v.sort_tensor, cur_group)
                    
                def f2(v):
                    cur_group_stack = self.smart_sortX(self.sort(v.cur_group.stack(), 1))
                    jj = tf.constant(0)
                    
                    def body(jj, cur_group_stack, v):
                        sort_tensor = v.sort_tensor.write(v.c, cur_group_stack[jj])
                        c = tf.add(v.c, 1)
                        return ALL_VARS(c, v.j, v.begin, v.inter_detection, sort_tensor, v.cur_group)
                        
                    _, _, v = tf.while_loop(lambda jj, cur_group_stack, v: tf.less(jj, v.j+1),
                                 lambda jj, cur_group_stack, v: (tf.add(jj,1), cur_group_stack, body(jj, cur_group_stack, v),),
                                 [jj, cur_group_stack, v])
                    cur_group = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                    j = tf.constant(0)
                    cur_group = cur_group.write(j, tensor[i])
                    begin = tf.constant(False)
                    
                    return ALL_VARS(v.c, j, begin, v.inter_detection, v.sort_tensor, cur_group)
                
                v = tf.cond(v.inter_detection, 
                       lambda: f1(v),
                       lambda: f2(v))
                inter_detection = tf.constant(False)
                
                return ALL_VARS(v.c, v.j, v.begin, inter_detection, v.sort_tensor, v.cur_group)
                
                
            all_vars = tf.cond(all_vars.begin, 
                    lambda: begin_condition(i, tensor, all_vars), 
                    lambda: other_condition(i, tensor, all_vars))
            
            return all_vars
                
        _, _, all_vars = tf.while_loop(lambda i, tensor, v: tf.less(i, tf.shape(tensor)[0]), 
                    lambda i, tensor, v: (tf.add(i,1), tensor, loop_body(i,tensor,v),), 
                    [i, tensor, ALL_VARS(c, j, begin, inter_detection, sort_tensor, cur_group)])
        
        def last_group(v):
            cur_group_stack = self.smart_sortX(self.sort(v.cur_group.stack(), 1))
            jj = tf.constant(0)
            
            def body(jj, cur_group_stack, v):
                sort_tensor = v.sort_tensor.write(v.c, cur_group_stack[jj])
                c = tf.add(v.c, 1)
                
                return ALL_VARS(c, v.j, v.begin, v.inter_detection, sort_tensor, v.cur_group)
            _, _, v = tf.while_loop(lambda jj, cur_group_stack, v: tf.less(jj, v.j+1),
                                          lambda jj, cur_group_stack, v: (tf.add(jj,1), cur_group_stack, body(jj, cur_group_stack, v),),
                                          [jj, cur_group_stack, v])
            return v
        
        def no_last_group(v):
            return v
        
        all_vars = tf.cond(all_vars.cur_group.size() != 0, lambda: last_group(all_vars), lambda: no_last_group(all_vars))
        
        return all_vars.sort_tensor.stack()
        

    @staticmethod
    def get_image_by_coordinates(image, coordinates):
        y_min, x_min, y_max, x_max = tf.unstack(tf.cast(coordinates, dtype=tf.int32))
        return image[y_min:y_max, x_min:x_max, :]

    @staticmethod
    def pad_word_img(sym, max_in_dims, constant_values):
        s = tf.shape(sym)
        paddings = [[tf.cast(tf.floor((max_in_dims[0] - s[0]) / 2), dtype=tf.int32),
                     max_in_dims[0] - s[0] - tf.cast(tf.floor((max_in_dims[0] - s[0]) / 2), dtype=tf.int32)],
                    [0, max_in_dims[1] - s[1]],
                    [0, 0]
                    ]
        return tf.pad(sym, paddings, 'CONSTANT', constant_values=constant_values)
    
    def pos2coords(self, pos, image):
        coords = tf.cond(tf.greater_equal(pos[0], tf.shape(image)[-2]),
                        lambda: tf.cast(tf.stack([0, 0, tf.shape(image)[-3], tf.shape(image)[-2]]), tf.float32),
                        lambda: tf.cond(tf.greater(pos[1], 0),
                                       lambda: tf.cond(tf.logical_and(tf.greater(pos[0], 4), tf.less(pos[1], tf.shape(image)[-2] - 4)),
                                                      lambda: tf.cast(tf.stack([0, pos[0], tf.shape(image)[-3], pos[1]]), tf.float32),
                                                      lambda: tf.cast(tf.stack([0, pos[0], tf.shape(image)[-3], pos[1]]), tf.float32)),
                                       lambda: tf.cast(tf.stack([0, pos[0], tf.shape(image)[-3], tf.shape(image)[-2]]), tf.float32)))
        return coords

    def recognize_word(self, image, coordinates):
        word_img = self.get_image_by_coordinates(image, coordinates)
        coef = tf.cast(tf.shape(word_img)[1], tf.float32)/tf.cast(self.ocr_word_width, tf.float32)
        coef_2 = tf.cast(tf.shape(word_img)[0], tf.float32)/tf.cast(self.ocr_word_height, tf.float32)
        
        entity_cond = tf.logical_and(tf.logical_and(tf.greater(tf.shape(word_img)[0], 3),
                                    tf.greater(tf.shape(word_img)[1], 3)),
                                    tf.logical_and(tf.greater(tf.cast(tf.shape(word_img)[0],tf.float32)/coef, 1),
                                    tf.greater(tf.cast(tf.shape(word_img)[0], tf.float32)/coef_2, 1)))
        def entity_word(word_img):
            word_img = tf.image.rgb_to_grayscale(word_img)
            word_img = tf.image.resize(word_img,
                                       [self.ocr_word_height, self.ocr_word_width],
                                       preserve_aspect_ratio=True)
            word_img = self.pad_word_img(word_img, [self.ocr_word_height, self.ocr_word_width, 3], 255)
            word_img = tf.io.encode_jpeg(tf.cast(word_img, dtype=tf.uint8))

            ocr_words_func = self.words_ocr_model.signatures['serving_default']
            return ocr_words_func(tf.expand_dims(word_img, axis=0))["output"]

        def no_entity_word():
            return tf.constant('#')
        
        word = tf.cond(entity_cond, lambda: entity_word(word_img), no_entity_word)
        
        return word

    def detect_words(self, image, coordinates, pair, i):
        line_img = self.get_image_by_coordinates(image, coordinates)
        line2word_func = self.words_detection_model.signatures['serving_default']
        positions = line2word_func(line_img)['output_0']

        words_coordinates = tf.map_fn(lambda pos: self.pos2coords(pos, line_img),
                                     positions,
                                     fn_output_signature=tf.float32,
                                     name='pos2coords')

        words_coordinates = self.sort(words_coordinates, 1)
        
        _, _, _, boxes = tf.while_loop(
            lambda num, words_coordinates, coordinates, boxes: tf.less(num, tf.shape(words_coordinates)[0]),
            lambda num, words_coordinates, coordinates, boxes: (tf.add(num, 1),
                                                                words_coordinates,
                                                               coordinates,
                                                               boxes.write(tf.cond(boxes.size() == 0, lambda: 0, lambda: num+boxes.size()-1), 
                                                                           tf.cast(tf.stack([coordinates[0], 
                                                                  coordinates[1] + words_coordinates[num,1], 
                                                                  coordinates[2],
                                                                 coordinates[1] + words_coordinates[num,3]]), dtype=tf.int32)),),
            [tf.constant(0), words_coordinates, coordinates, pair.boxes])
        
        words = tf.map_fn(
            lambda coord: self.recognize_word(line_img, coord),
            words_coordinates,
            fn_output_signature=tf.string,
            name="words_loop"
        )
        return self.OUTPUT(pair.text.write(i, tf.strings.reduce_join(words, separator=" ")), boxes)

    def detect_lines(self, image, boxes, scores):
        lines = self.get_coordinates(boxes, scores, tf.shape(image), self.lines_detection_threshold)
        lines = self.sort(self.sort(lines, 1), 0)
        lines = self.smart_sortY(lines)
        
        boxes = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False, element_shape=(4,))
        full_text = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        
        _, _, _, out = tf.while_loop(lambda num, lines, image, p: tf.less(num, tf.shape(lines)[0]),
                     lambda num, lines, image, p: (tf.add(num,1),
                                                  lines,
                                                  image,
                                                  self.detect_words(image, lines[num], p, num),),
                     [tf.constant(0), lines, image, self.OUTPUT(full_text, boxes)])
        return tf.strings.reduce_join(out.text.stack(), separator="\n"), out.boxes.stack()

    def thumbnail_global(self, img):
        if tf.shape(img)[-2] >= tf.shape(img)[-3]:
            img = tf.cond(tf.cast(tf.shape(img)[-2] <= self.max_image_size, tf.bool), 
                          lambda: tf.cast(img, dtype=tf.uint8), lambda: tf.cast(tf.image.resize(img, 
                    [tf.cast(tf.cast(tf.shape(img)[-3], tf.float32)/(tf.cast(tf.shape(img)[-2]/self.max_image_size, 
                                                                             tf.float32)), tf.int32), self.max_image_size]), dtype=tf.uint8))
        else:
            img = tf.cond(tf.cast(tf.shape(img)[-3] <= self.max_image_size, tf.bool), 
                          lambda: tf.cast(img, dtype=tf.uint8), lambda: tf.cast(tf.image.resize(img, 
                    [self.max_image_size, tf.cast(tf.cast(tf.shape(img)[-2], tf.float32)/(tf.cast(tf.shape(img)[-3]/self.max_image_size, 
                                                                             tf.float32)), tf.int32)]), dtype=tf.uint8))
        return img

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def detect(self, string_inp):
        images = tf.map_fn(
            lambda img: tf.io.decode_image(img, expand_animations=True),
            string_inp,
            fn_output_signature=tf.uint8
        )
        images.set_shape((None, None, None, None))

        detect_lines_func = self.lines_detection_model.signatures['serving_default']
        lines_detections = detect_lines_func(images)

        return tf.map_fn(
            lambda args: self.detect_lines(args[0], args[1], args[2]),
            (images, lines_detections['detection_boxes'], lines_detections['detection_scores']),
            fn_output_signature=(tf.string, tf.int32),
            name="batches_loop"
        )
    
if __name__ == '__main__':
    k.backend.clear_session()
    model = Image2Text(
        "models/lines_detector/1.2",
        "models/words_detector/graph",
        "models/words_ocr/1.4"
    )
    tf.saved_model.save(model, 'ScanToText')
