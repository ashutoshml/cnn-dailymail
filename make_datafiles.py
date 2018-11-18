import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# src_tokenized_file = "src_tokenized_file.txt"
# tgt_tokenized_file = "tgt_tokenized_file.txt"
finished_files_dir = "finished_files"
tokenized_files_dir = "tokenized_files"

VOCAB_SIZE = 30000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(set_name, finished_files_dir):
    in_file = '{}/{}.bin'.format(finished_files_dir, set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    chunks_dir = os.path.join(finished_files_dir, "chunked")
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1
    reader.close()


def chunk_all(finished_files_dir):
    # Make a dir to hold the chunks
    chunks_dir = os.path.join(finished_files_dir, "chunked")
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'valid', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name, finished_files_dir)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_sentences(sentence_file, sentence_tokenized_file):
    print("Preparing to tokenize %s to %s..." % (sentence_file, sentence_tokenized_file))
    # make IO list file
    with open('mapping.txt', 'w') as f:
        f.write('{} \t {}\n'.format(sentence_file, sentence_tokenized_file))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing sentences in %s and saving in %s..." % (sentence_file, sentence_tokenized_file))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove('mapping.txt')

    # Check that the tokenized stories directory contains the same number of files as the original directory
    print("Successfully finished tokenizing {} to {}".format(sentence_file, sentence_tokenized_file))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def get_sent_bin(src_file, tgt_file):
    src_lines = read_text_file(src_file)
    tgt_lines = read_text_file(tgt_file)

    # Lowercase everything
    src_lines = [line.lower() for line in src_lines]
    tgt_lines = [line.lower() for line in tgt_lines]

    return src_lines, tgt_lines


def write_to_bin(src_tokenized_file, tgt_tokenized_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the sentence_tokenized_file and writes them to a out_file."""
    print("Making bin file for URLs listed in %s..." % src_tokenized_file)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        if os.path.isfile(os.path.join(src_tokenized_file)):
            src_file = os.path.join(src_tokenized_file)

        if os.path.isfile(os.path.join(tgt_tokenized_file)):
            tgt_file = os.path.join(tgt_tokenized_file)
        # Get the strings to write to .bin file
        srcs, tgts = get_sent_bin(src_file, tgt_file)
        print('____________________________________')
        print(len(srcs))
        print(len(tgts))
        print('____________________________________')

        for src, tgt in zip(srcs, tgts):
            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['src'].bytes_list.value.extend([src.encode()])
            tf_example.features.feature['tgt'].bytes_list.value.extend([tgt.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                src_tokens = src.split()
                tgt_tokens = tgt.split()
                tgt_tokens = [t for t in tgt_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = src_tokens + tgt_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python make_datafiles.py <data_path> <dataset>")
        sys.exit()
    # src = sys.argv[1]
    # tgt = sys.argv[2]
    data_path = sys.argv[1]
    dataset = sys.argv[2]

    finished_files_dir = os.path.join(finished_files_dir, dataset)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)
    tokenized_files_dir = os.path.join(tokenized_files_dir, dataset)
    if not os.path.exists(tokenized_files_dir):
        os.makedirs(tokenized_files_dir)
    # Run stanford tokenizer on both src and tgt, outputting to tokenized file.txt

    src = os.path.join(data_path, 'train_src.txt')
    tgt = os.path.join(data_path, 'train_tgt.txt')
    src_train_tokenized_file = os.path.join(tokenized_files_dir, 'train_src_tokenized.txt')
    tgt_train_tokenized_file = os.path.join(tokenized_files_dir, 'train_tgt_tokenized.txt')
    tokenize_sentences(src, src_train_tokenized_file)
    tokenize_sentences(tgt, tgt_train_tokenized_file)

    src = os.path.join(data_path, 'val_src.txt')
    tgt = os.path.join(data_path, 'val_tgt.txt')
    src_val_tokenized_file = os.path.join(tokenized_files_dir, 'val_src_tokenized.txt')
    tgt_val_tokenized_file = os.path.join(tokenized_files_dir, 'val_tgt_tokenized.txt')
    tokenize_sentences(src, src_val_tokenized_file)
    tokenize_sentences(tgt, tgt_val_tokenized_file)

    src = os.path.join(data_path, 'test_src.txt')
    tgt = os.path.join(data_path, 'test_tgt.txt')
    src_test_tokenized_file = os.path.join(tokenized_files_dir, 'test_src_tokenized.txt')
    tgt_test_tokenized_file = os.path.join(tokenized_files_dir, 'test_tgt_tokenized.txt')
    tokenize_sentences(src, src_test_tokenized_file)
    tokenize_sentences(tgt, tgt_test_tokenized_file)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    # write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
    # write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(src_train_tokenized_file, tgt_train_tokenized_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
    write_to_bin(src_val_tokenized_file, tgt_val_tokenized_file, os.path.join(finished_files_dir, "valid.bin"))
    write_to_bin(src_test_tokenized_file, tgt_test_tokenized_file, os.path.join(finished_files_dir, "test.bin"))

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all(finished_files_dir)
