#coding: utf-8
import tensorflow as tf

# csv line to tensor x, y
def parse_csv_line(line_str, n_fields = 9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line_str, record_defaults = defs)  # n_fields个tensor的数组
    x = tf.stack(parsed_fields[:-1]) # 具有n_fields -1 个元素的tensor
    y = tf.stack(parsed_fields[-1:])
    return x, y

# csv files to dataset
def csv_reader_dataset(file_names, n_readers= 5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(file_names)
    dataset = dataset.repeat() # unlimit repeat
    dataset = dataset.interleave(
        lambda fn: tf.data.TextLineDataset(fn).skip(1),
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(
        parse_csv_line,
        num_parallel_calls = n_parse_threads
    )
    dataset = dataset.batch(batch_size)
    return dataset

# write dataset to csv files
def save_to_csv(output_dir, data, name_prefix, header=None, n_parts = 10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    file_names = []
    
    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        file_names.append(part_csv)
        with open(part_csv, 'wt', encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join(repr(col_t) for col_t in data[row_index]))
                f.write("\n")
    return file_names

def serialize_example(x, y):
    """Converts x, y to tf.train.Example and serialize"""
    input_features = tf.train.FloatList(value = x)
    label = tf.train.FloatList(value = y)
    features = tf.train.Features(
        feature = {
            "input_features":tf.train.Feature(float_list = input_features),
            "label": tf.train.Feature(float_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

  def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    display(Markdown('```python\n{}\n```'.format(code)))

import unicodedata
def unicode_to_ascii(s):
    # NFD 是normalize的一种方法，作用是如果一个unicode是多个ascii组成的，就把其拆开
    # Mn 注音
    return ' '.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
