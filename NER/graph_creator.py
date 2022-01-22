import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import pickle
import json
import collections

with open('objects/word2idx.pickle', 'rb') as f:
    data = pickle.load(f)
with open('objects/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

idx2tag = {0:'OTHER',1:'B_ADDRESS',2:'I_ADDRESS',3:'B_LEGALNAME',4:'I_LEGALNAME',
          5:'B_NAME',6:'I_NAME',7:'B_REGIONS',8:'I_REGIONS',9:'B_INN',10:'I_INN',
          11:'B_KPP',12:'I_KPP',13:'B_OGRN',14:'I_OGRN',15:'B_REGCBR',16:'I_REGCBR',
          17:'B_PHONE',18:'I_PHONE',19:'B_EMAIL',20:'I_EMAIL',21:'PAD'}

tag2class = {'B_ADDRESS' : 0, 'I_ADDRESS' : 0, 'B_LEGALNAME' : 1, 'I_LEGALNAME' : 1,
            'B_NAME' : 2, 'I_NAME' : 2, 'B_REGIONS' : 3, 'I_REGIONS' : 3,
            'B_INN' : 4, 'I_INN' : 4, 'B_KPP' : 5, 'I_KPP' : 5,
            'B_OGRN' : 6, 'I_OGRN' : 6, 'B_REGCBR' : 7, 'I_REGCBR' : 7,
            'B_PHONE' : 8, 'I_PHONE' : 8, 'B_EMAIL' : 9, 'I_EMAIL' : 9}

dct_words = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(data.keys())),
        values=tf.constant(np.arange(len(data), dtype=np.float32), dtype=tf.float32),
    ),
    default_value=tf.constant(1, dtype=tf.float32),)
dct_shorts = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(config['shorts'].keys())),
        values=tf.constant(list(config['shorts'].values())),
    ),
    default_value=tf.constant('UNK'),)
dct_tags = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(idx2tag.keys())),
        values=tf.constant(list(idx2tag.values())),
    ),
    default_value=tf.constant('UNK'),)
dct_class = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(tag2class.keys())),
        values=tf.constant(list(tag2class.values())),
    ),
    default_value=tf.constant(-1),)
alphabet = "абвгдеёжзийклмнопрстуфхцчшщьыъэюяabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

dct_alpha = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(['UNK', 'PAD'] + [l for l in alphabet]),
        values=tf.constant(np.arange(len([l for l in alphabet]) + 2, dtype=np.float32), dtype=tf.float32),
    ),
    default_value=tf.constant(0, dtype=tf.float32),)

class crf_loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss, _ = tfa.text.crf_log_likelihood(
            y_pred,
            y_true,
            tf.map_fn(lambda x: tf.shape(x)[0], y_pred, fn_output_signature=tf.int32),
            transition_params
        )
        
        return -tf.reduce_mean(loss)

class NER(tf.Module):
    def __init__(self, dct_shorts, dct_alpha, dct_words, dct_tags, dct_classes, model_path):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'crf_loss':crf_loss})
        self.dct_alpha = dct_alpha
        self.dct_words = dct_words
        self.dct_shorts = dct_shorts
        self.dct_tags = dct_tags
        self.dct_classes = dct_classes
        self.MAX_WORD_LEN = 50
        self.MAX_SENT_LEN = 100
        
    def feature_gen(self, word):
        REG_PHONE = r"^[\d|\s|\+|\(|\)|\-|\,|\;|\!|\?]+$"
        REG_CBR = r"^\d{13}$|^\d{15}$|^№\d{13}$|^№\d{15}$"
        OGRN = r"^\d{13}$"
        KPP = r"^\d{9}$"
        INN = r"^\d{10}$|^\d{12}$|^\d{10}/$|^\d{12}/$"

        username_re=r"(?P<username>[\w][\w_.-]*)"
        domain_re=r"(?P<domain>[\w][\w_.-]*)"
        zone_re=r"(?P<zone>[a-z]{2}|aero|asia|biz|cat|com|coop|edu|gov|info|int|jobs|mil|moby|museum|name|net|org|pro|tel|travel|xxx)"

        REG_EMAIL = r"(?P<space>(\s|%20|\b)){}@{}dot{}\b".format(username_re, domain_re, zone_re)
        REG_DATE = r"^\d{,2}dot\d{,2}dot\d{,4}"

        regulars = tf.constant([REG_PHONE,REG_CBR,OGRN,KPP,INN,REG_EMAIL,REG_DATE])

        feature = tf.map_fn(lambda reg: tf.cast(tf.strings.regex_full_match(word, reg), tf.float32),
                           regulars,
                           fn_output_signature=tf.float32)
        feature = tf.cond(tf.reduce_sum(feature) > 0,
                         lambda: tf.concat([feature, tf.constant([1.])], 0),
                         lambda: tf.concat([feature, tf.constant([0.])], 0))
        return feature
        
    def replace_short(self, txt):
        
        def f(word):
            word = tf.cond(self.dct_shorts.lookup(tf.strings.regex_replace(word, '[.]', '')) == 'UNK',
                          lambda: word,
                          lambda: self.dct_shorts.lookup(tf.strings.regex_replace(word, '[.]', '')))
            word = tf.cond(tf.strings.regex_full_match(word, '.*[.].*[а-яa-z]'),
                          lambda: tf.strings.regex_replace(word, '[.]', 'dot'),
                          lambda: word)
            return word
        
        txt = tf.map_fn(lambda w: f(w),
                 txt,
                 fn_output_signature=tf.string)
        return tf.strings.reduce_join(txt, separator=' ')
    
    def prepare_data(self, txt, orig_txt):
        
        orig = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        proc = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        
        TEXTS = collections.namedtuple('TEXTS', 'proc, orig, pos')
        
        def body(p, sent, txt, i):
            sent = tf.strings.split(sent, ' ')
            proc = p.proc.write(i, sent)
            
            cur_pos = tf.shape(sent)[0]
            pos = p.pos + cur_pos
            
            orig = p.orig.write(i, tf.strings.split(txt, ' ')[p.pos:pos])
            
            return TEXTS(proc, orig, pos)
            
        txt = tf.strings.split(txt, '. ')
        
        i = tf.constant(0)
        
        _, _, pair = tf.while_loop(lambda i, txt, pair: tf.less(i, tf.shape(txt)[0]),
                                     lambda i, txt, pair: (tf.add(i,1),
                                                           txt,
                                                          body(pair, txt[i], orig_txt, i)),
                                  [i, txt, TEXTS(proc, orig, tf.constant(0))])
        
        
        orig_limit = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        proc_limit = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        
        TEXTS_LIMIT = collections.namedtuple('TEXTS_LIMIT', 'proc_limit, orig_limit, j')
        
        def body_limit(pair, pair_limit, i):
            cur_proc = pair.proc.read(i)
            cur_orig = pair.orig.read(i)
            
            j = pair_limit.j
            
            def f(cur_proc, cur_orig, pair_limit):
                steps = tf.shape(cur_proc)[0] // self.MAX_SENT_LEN + 1
                
                n = tf.constant(0)
                proc = pair_limit.proc_limit
                orig = pair_limit.orig_limit
                j = pair_limit.j
                
                _, j, proc, orig = tf.while_loop(lambda n, j, proc, orig: tf.less(n, steps),
                             lambda n, j, proc, orig: (tf.add(n,1),
                                           tf.add(j,1),
                                          proc.write(j, cur_proc[n*self.MAX_SENT_LEN: (n+1)*self.MAX_SENT_LEN]),
                                          orig.write(j, cur_orig[n*self.MAX_SENT_LEN: (n+1)*self.MAX_SENT_LEN])),
                                        [n, j, proc, orig])
                return proc, orig, j
            
            proc_limit, orig_limit, j = tf.cond(tf.less(tf.shape(cur_proc), self.MAX_SENT_LEN),
                                            lambda: (pair_limit.proc_limit.write(j, cur_proc), 
                                                     pair_limit.orig_limit.write(j, cur_orig),
                                                     pair_limit.j + 1),
                                            lambda: f(cur_proc, cur_orig, pair_limit))
            return TEXTS_LIMIT(proc_limit, orig_limit, j)
        
        i = tf.constant(0)
        
        _, _, pair_limit = tf.while_loop(lambda i, pair, pair_limit: tf.less(i, pair.proc.size()),
                                         lambda i, pair, pair_limit: (tf.add(i,1),
                                                                     pair,
                                                                     body_limit(pair, pair_limit, i)),
                                        [i, pair, TEXTS_LIMIT(proc_limit, orig_limit, tf.constant(0))])
        
        ORIG = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        WORDS = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
        
        CHARS = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        FEATS = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        
        INPUT = collections.namedtuple('INPUT', 'orig, words, chars, feats, pos')
        
        def main_body(i, p, inp):
            cur_proc = p.proc_limit.read(i)
            cur_orig = p.orig_limit.read(i)
            
            orig = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)
            words = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False)

            chars = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            feats = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            
            input = collections.namedtuple('input', 'orig, words, chars, feats, pos')
            
            def inner_body(j, proc, orig, inp):
                word = proc[j]
                word_orig = orig[j]
                
                def f(inp, word, word_orig, j):
                    char_w = self.dct_alpha.lookup(tf.strings.unicode_split(word, input_encoding='UTF-8'))[:self.MAX_WORD_LEN]
                    char_w = tf.pad(char_w, [[0,self.MAX_WORD_LEN - tf.shape(char_w)[0]]], constant_values=0.)
                    
                    feat = self.feature_gen(word)
                    
                    dig_cond = tf.logical_and(tf.strings.length(tf.strings.regex_replace(word, 
                            r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^_`{|}~0-9]', '')) == 0, 
                                             tf.strings.length(tf.strings.regex_replace(word, 
                            r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^_`{|}~]', '')) != 0)
                    
                    word = tf.cond(dig_cond,
                                  lambda: tf.constant('цифры'),
                                  lambda: tf.strings.regex_replace(word, r'[^a-zA-Zа-яА-Яё@]+', ''))
                    
                    orig, words, chars, feats, pos = tf.cond(tf.strings.length(word) > 0,
                                                       lambda: (inp.orig.write(inp.pos, word_orig),
                                                               inp.words.write(inp.pos, word),
                                                               inp.chars.write(inp.pos, char_w),
                                                               inp.feats.write(inp.pos, feat),
                                                               tf.add(inp.pos,1)),
                                                       lambda: (inp.orig, inp.words, inp.chars, inp.feats, inp.pos))
                    return orig, words, chars, feats, pos
                    
                    
                
                orig, words, chars, feats, pos = tf.cond(tf.strings.length(tf.strings.regex_replace(
                    word, r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^_`{|}~]', '')) == 0,
                       lambda: (inp.orig, inp.words, inp.chars, inp.feats, inp.pos),
                       lambda: f(inp, word, word_orig, j))
                
                return input(orig, words, chars, feats, pos)
            
            _, _, _, inp_w = tf.while_loop(lambda j, proc, orig, inp: tf.less(j, tf.shape(proc)[0]),
                         lambda j, proc, orig, inp: (tf.add(j,1),
                                                    proc,
                                                    orig,
                                                    inner_body(j, proc, orig, inp)),
                                        [tf.constant(0), cur_proc, cur_orig, input(orig, words, chars, feats, tf.constant(0))])
            
            def f(inp_w, inp):
                orig = inp_w.orig.stack()
                orig = tf.pad(orig,  [[0, self.MAX_SENT_LEN - tf.shape(orig)[0]]], constant_values=b'PAD')

                words = inp_w.words.stack()
                words = tf.pad(words,  [[0, self.MAX_SENT_LEN - tf.shape(words)[0]]], constant_values=b'PAD')

                chars = inp_w.chars.stack()
                chars = tf.pad(chars,  [[0, self.MAX_SENT_LEN - tf.shape(chars)[0]], [0,0]], constant_values=tf.constant(0.))

                feats = inp_w.feats.stack()
                feats = tf.pad(feats,  [[0, self.MAX_SENT_LEN - tf.shape(feats)[0]], [0,0]], constant_values=0.)

                ORIG = inp.orig.write(inp.pos, orig)
                WORDS = inp.words.write(inp.pos, words)
                CHARS = inp.chars.write(inp.pos, chars)
                FEATS = inp.feats.write(inp.pos, feats)
                POS = tf.add(inp.pos, 1)
                
                return ORIG, WORDS, CHARS, FEATS, POS
                
            
            ORIG, WORDS, CHARS, FEATS, POS = tf.cond(inp_w.words.size() > 0,
                                                    lambda: f(inp_w, inp),
                                                    lambda: (inp.orig,
                                                            inp.words,
                                                            inp.chars,
                                                            inp.feats,
                                                            inp.pos))
            
            return INPUT(ORIG, WORDS, CHARS, FEATS, POS)
            
        
        _, _, inp = tf.while_loop(lambda i, p, inp: tf.less(i, p.proc_limit.size()),
                     lambda i, p, inp: (tf.add(i,1),
                                        p,
                                        main_body(i, p, inp)),
                                 [tf.constant(0), pair_limit, INPUT(ORIG, WORDS, CHARS, FEATS, tf.constant(0))])
        
        return inp.orig.stack(), self.dct_words.lookup(inp.words.stack()), inp.chars.stack(), inp.feats.stack()
    
    def predict(self, orig, w, c, f):
        
        output = tf.cast(tf.argmax(self.model([tf.reshape(c, [-1, self.MAX_SENT_LEN*self.MAX_WORD_LEN]), w, f]), -1), tf.int32)
        
        final = tf.constant(['Address: ', 'Legal name: ', 'Name: ', 'Regions: ', 'INN: ', 'KPP: ', 'OGRN: ', 'REGCBR: ',
                            'Phone: ', 'E-mail: '])

        def main_body(i, orig, output, final):
            cur_orig = orig[i]
            cur_out = output[i]
            
            def inner_body(j, orig, out, final):
                
                entity_cond = tf.logical_and(out[j] != 0, out[j] != 21)
                
                def f(pos, ent, final, orig, j):
                    ent = tf.cond(final[pos] == ent,
                                 lambda: ent + orig[j] + ' ',
                                 lambda: ent)
                    return ent
                
                final = tf.cond(entity_cond,
                       lambda: tf.map_fn(lambda e: f(self.dct_classes.lookup(self.dct_tags.lookup(out[j])),
                                                    e,
                                                    final,
                                                    orig,
                                                    j),
                                        final,
                                        fn_output_signature=tf.string),
                       lambda: final)
                
                return final
            
            _, _, _, final = tf.while_loop(lambda j, cur_orig, cur_out, final: tf.less(j, tf.shape(cur_orig)[0]),
                                          lambda j, cur_orig, cur_out, final: (tf.add(j, 1),
                                                                              cur_orig,
                                                                              cur_out,
                                                                              inner_body(j, cur_orig, cur_out, final)),
                                          [tf.constant(0), cur_orig, cur_out, final])
            return final
        
        _, _, _, final = tf.while_loop(
            lambda i, orig, output, final: tf.less(i, tf.shape(output)[0]),
            lambda i, orig, output, final: (
                tf.add(i,1),
                orig,
                output,
                main_body(i, orig, output, final)
            ),
            [tf.constant(0), orig, output, final]
        )
        
        print(final)
        
        return final
        
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def call(self, string_inp):
        cur_text = tf.map_fn(lambda s: tf.io.read_file(s),
                         string_inp,
                         fn_output_signature=tf.string)
        cur_text = tf.map_fn(lambda s: tf.strings.regex_replace(s, r'[\t\n\v\f\r\xa0]', ''),
                         cur_text,
                         fn_output_signature=tf.string)
        cur_text = tf.map_fn(lambda s: tf.strings.regex_replace(s, r'[,]', ' , '),
                         cur_text,
                         fn_output_signature=tf.string)
        cur_text = tf.map_fn(lambda s: tf.strings.regex_replace(s, r'[/]', ' / '),
                         cur_text,
                         fn_output_signature=tf.string)
        orig_text = tf.map_fn(lambda s: tf.strings.regex_replace(s, 
                                                                r'[!?;]', '. '),
                         cur_text,
                         fn_output_signature=tf.string)
        proc_text = tf.map_fn(lambda s: self.replace_short(tf.strings.split(tf.strings.lower(s, encoding='utf-8'), ' ')),
                         orig_text,
                         fn_output_signature=tf.string)
        orig, x_words, x_chars, x_feats = tf.map_fn(lambda args: self.prepare_data(args[0], args[1]),
                        (proc_text, orig_text),
                        fn_output_signature=(tf.string, tf.float32, tf.float32, tf.float32))
        answ = tf.map_fn(lambda args: self.predict(args[0], args[1], args[2], args[3]),
                        (orig, x_words, x_chars, x_feats),
                        fn_output_signature=tf.string)
        
        def write2file(inp, answ):
            
            out_file = tf.strings.regex_replace(inp, '[.*].*', '') + tf.constant('_result.txt')
        
            content = tf.strings.reduce_join(answ, separator='\n')
            
            tf.io.write_file(out_file, content)
            
            return out_file
        
        _ = tf.map_fn(lambda args: write2file(args[0], args[1]),
                     (string_inp, answ),
                     fn_output_signature=tf.string)
        
        return answ
    
if __name__ == '__main__':
    tf.keras.backend.clear_session()
    model = NER(dct_shorts, dct_alpha, dct_words, dct_tags, dct_class, 'objects/ner_model.h5')
    tf.saved_model.save(model, 'tf_graph')