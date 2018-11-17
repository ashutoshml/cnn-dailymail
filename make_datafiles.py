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

# all_train_urls = "url_lists/all_train.txt"
# all_val_urls = "url_lists/all_val.txt"
# all_test_urls = "url_lists/all_test.txt"

src_tokenized_file = "src_tokenized_file"
tgt_tokenized_file = "tgt_tokenized_file"
finished_files_dir = "finished_files"

# These are the number of .story files we expect there to be in src and tgt
# num_expected_cnn_stories = 92579
# num_expected_dm_stories = 219506

VOCAB_SIZE = 30000
CHUNK_SIZE = 100 # num examples per chunk, for the chunked data


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
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name, finished_files_dir)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_sentences(sentence_file, sentence_tokenized_file):
    print("Preparing to tokenize %s to %s..." % (sentence_file, sentence_tokenized_file))
    # make IO list file
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines']
    command.append(sentence_file)
    print("Tokenizing sentences in %s and saving in %s..." % (sentence_file, sentence_tokenized_file))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    print("Successfully finished tokenizing %s to %s.\n" % (sentence_file, sentence_tokenized_file))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def get_sent_bin(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Separate out src and tgt sentences
    src_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        else:
            src_lines.append(line)

    # Make src into a single string
    src = ' '.join(src_lines)

    # Make tgt into a signle string, putting <s> and </s> tags around the sentences
    tgt = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return src, tgt


def write_to_bin(src_tokenized_file, tgt_tokenized_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the sentence_tokenized_file and writes them to a out_file."""
    print("Making bin file for URLs listed in %s..." % sentence_tokenized_file)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        if os.path.isfile(os.path.join(src_tokenized_file)):
            src_file = os.path.join(src_tokenized_file)

        if os.path.isfile(os.path.join(tgt_tokenized_file)):
            tgt_file = os.path.join(tgt_tokenized_file)
        # Get the strings to write to .bin file
        src = get_sent_bin(src_file)
        tgt = get_sent_bin(tgt_file)

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
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("USAGE: python make_datafiles.py <src> <tgt> <dataset>")
        sys.exit()
    src = sys.argv[1]
    tgt = sys.argv[2]
    dataset = sys.argv[3]

    # Check the stories directories contain the correct number of .story files
    # check_num_stories(src, num_expected_cnn_stories)
    # check_num_stories(tgt, num_expected_dm_stories)

    # Create some new directories
    if not os.path.exists(src_tokenized_file): os.makedirs(src_tokenized_file)
    if not os.path.exists(tgt_tokenized_file): os.makedirs(tgt_tokenized_file)

    finished_files_dir = os.path.join(finished_files_dir, dataset)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_sentences(src, src_tokenized_file)
    tokenize_sentences(tgt, tgt_tokenized_file)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(src_tokenized_file, tgt_tokenized_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all(finished_files_dir)
