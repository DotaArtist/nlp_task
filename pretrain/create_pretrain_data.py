from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import sentencepiece as spm

SPIECE_UNDERLINE = u"▁".encode("utf-8")

def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def encode_ids(sp_model, text, sample=False):
  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return six.ensure_text(text, "utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return six.ensure_text(text, "utf-8", "ignore")
    elif isinstance(text, six.text_type):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

class tokenization(object):
    def printable_text(text):
      """Returns text encoded in a way suitable for print or `tf.logging`."""

      # These functions want `str` for both Python2 and Python3, but in one case
      # it's a Unicode string and in the other it's a byte string.
      if six.PY3:
        if isinstance(text, str):
          return text
        elif isinstance(text, bytes):
          return six.ensure_text(text, "utf-8", "ignore")
        else:
          raise ValueError("Unsupported string type: %s" % (type(text)))
      elif six.PY2:
        if isinstance(text, str):
          return text
        elif isinstance(text, six.text_type):
          return six.ensure_binary(text, "utf-8")
        else:
          raise ValueError("Unsupported string type: %s" % (type(text)))
      else:
        raise ValueError("Not running on Python2 or Python 3?")

printable_text = tokenization.printable_text


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip().split()[0] if token.strip() else " "
      if token not in vocab:
        vocab[token] = len(vocab)
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

def preprocess_text(inputs, remove_space=True, lower=False):
  """preprocess data by removing extra space and normalize data."""
  outputs = inputs
  if remove_space:
    outputs = " ".join(inputs.strip().split())

  if six.PY2 and isinstance(outputs, str):
    try:
      outputs = six.ensure_text(outputs, "utf-8")
    except UnicodeDecodeError:
      outputs = six.ensure_text(outputs, "latin-1")

  outputs = unicodedata.normalize("NFKD", outputs)
  outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  """turn sentences into word pieces."""

  if six.PY2 and isinstance(text, six.text_type):
    text = six.ensure_binary(text, "utf-8")

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
  new_pieces = []
  for piece in pieces:
    piece = printable_text(piece)
    if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = six.ensure_text(piece, "utf-8")
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + six.ensure_str(substr)
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True, spm_model_file=None):
        self.vocab = None
        self.sp_model = None
        if spm_model_file:
            self.sp_model = spm.SentencePieceProcessor()
            tf.logging.info("loading sentence piece model")
            # Handle cases where SP can't load the file, but gfile can.
            sp_model_ = tf.gfile.GFile(spm_model_file, "rb").read()
            self.sp_model.LoadFromSerializedProto(sp_model_)
            # Note(mingdachen): For the purpose of consisent API, we are
            # generating a vocabulary for the sentence piece tokenizer.
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                          in range(self.sp_model.GetPieceSize())}
        else:
            self.vocab = load_vocab(vocab_file)
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_scratch(cls, vocab_file, do_lower_case, spm_model_file):
        return FullTokenizer(vocab_file, do_lower_case, spm_model_file)

    @classmethod
    def from_hub_module(cls, hub_module, use_spm=True):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            albert_module = hub.Module(hub_module)
            tokenization_info = albert_module(signature="tokenization_info",
                                              as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run(
                    [tokenization_info["vocab_file"],
                     tokenization_info["do_lower_case"]])
        if use_spm:
            spm_model_file = vocab_file
            vocab_file = None
        return FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case,
            spm_model_file=spm_model_file)

    def tokenize(self, text):
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        if self.sp_model:
            tf.logging.info("using sentence piece tokenzier.")
            return [self.sp_model.PieceToId(
                printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        if self.sp_model:
            tf.logging.info("using sentence piece tokenzier.")
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            return convert_by_vocab(self.inv_vocab, ids)


import collections
import random

import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string("input_file_mode", "r",
                    "The data format of the input file.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", True,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_bool(
    "do_permutation", False,
    "Whether to do the permutation training.")

flags.DEFINE_bool(
    "favor_shorter_ngram", True,
    "Whether to set higher probabilities for sampling shorter ngrams.")

flags.DEFINE_bool(
    "random_next_sentence", False,
    "Whether to use the sentence that's right before the current sentence "
    "as the negative sample for next sentence prection, rather than using "
    "sentences from other random documents.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("ngram", 3, "Maximum number of ngrams to mask.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 40,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next, token_boundary):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.token_boundary = token_boundary
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "token_boundary: %s\n" % (" ".join(
            [str(x) for x in self.token_boundary]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        token_boundary = list(instance.token_boundary)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_boundary.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        multiplier = 1 + int(FLAGS.do_permutation)
        while len(masked_lm_positions) < max_predictions_per_seq * multiplier:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        sentence_order_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["token_boundary"] = create_int_feature(token_boundary)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        # Note: We keep this feature name `next_sentence_labels` to be compatible
        # with the original data created by lanzhzh@. However, in the ALBERT case
        # it does contain sentence_order_label.
        features["next_sentence_labels"] = create_int_feature(
            [sentence_order_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, FLAGS.input_file_mode) as reader:
            while True:
                line = reader.readline()
                if not FLAGS.spm_model_file:
                    line = convert_to_unicode(line)
                if not line:
                    break
                if FLAGS.spm_model_file:
                    line = preprocess_text(line, lower=FLAGS.do_lower_case)
                else:
                    line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or \
                        (FLAGS.random_next_sentence and rng.random() < 0.5):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif not FLAGS.random_next_sentence and rng.random() < 0.5:
                    is_random_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels, token_boundary) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    token_boundary=token_boundary,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def _is_start_piece_sp(piece):
    """Check if the current word piece is the starting piece (sentence piece)."""
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    special_pieces.add(u"€".encode("utf-8"))
    special_pieces.add(u"£".encode("utf-8"))
    # Note(mingdachen):
    # For foreign characters, we always treat them as a whole piece.
    english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
    if (six.ensure_str(piece).startswith("▁") or
            six.ensure_str(piece).startswith("<") or piece in special_pieces or
            not all([i.lower() in english_chars.union(special_pieces)
                     for i in piece])):
        return True
    else:
        return False


def _is_start_piece_bert(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece):
    if FLAGS.spm_model_file:
        return _is_start_piece_sp(piece)
    else:
        return _is_start_piece_bert(piece)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(token)):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(token):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, FLAGS.ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, FLAGS.ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not FLAGS.favor_shorter_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    rng.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                               pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    rng.shuffle(ngram_indexes)

    select_indexes = set()
    if FLAGS.do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                   pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.spm_model_file)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    tf.logging.info("number of instances: %i", len(instances))

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()

    #
    """bash
    python ./create_pretraining_data.py --input_file "d:/data_file/medical_data/medical_file_00000" --output_file "medical_data_train" --v
ocab_file "D:/model_file/my_albert/vocab.txt" --max_seq_length=512
    """
