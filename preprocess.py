import glob
import csv
import json
import os
import numpy as np

from music_dictionary import chord_dictionary, scale_dictionary, key_signature_calculator, root_list, chord_list, unused_note


def convert_chord_type(chord):
    """Change all the chord to the major or minor."""
    return chord_dictionary.get(chord, None)


def scale_to_integer(scale):
    """Convert the scale to integer."""
    return scale_dictionary.get(scale, None)


def get_transpose_interval(key):
    """Get the interval to transpose from key signature."""
    return key_signature_calculator.get(key, None)


def transpose(root, key):
    """Transpose the key of the chords."""
    root_integer = scale_to_integer(root)  # result : 0 ~ 11
    transpose_interval = get_transpose_interval(key)

    transposed_index = root_integer + transpose_interval
    if transposed_index >= 12:
        transposed_index = transposed_index - 12

    return root_list[transposed_index]


def preprocess(paths):
    vectorized_songs = []
    for path in glob.glob(paths):
        csv_file = open(path, 'r', encoding='utf-8')
        reader = csv.reader(csv_file)
        next(reader)  # skip first line

        song = dict()
        for line in reader:
            measure, key_fifths = line[1], line[2]
            chord_root, chord_type, note_root = line[4], line[5], line[6]

            # ignore unused note
            if chord_type in unused_note or note_root in unused_note:
                continue

            # convert all the chord to major or minor
            result_chord_type = convert_chord_type(chord_type)

            # transpose
            result_chord = transpose(chord_root, key_fifths) + ':' + result_chord_type
            result_note = transpose(note_root, key_fifths)

            if measure not in song:
                song[measure] = {'chord': result_chord, 'note_sequence': [result_note]}
            else:
                song.get(measure).get('note_sequence').append(result_note)
        csv_file.close()

        # vectorize the song
        inputs, targets = [], []
        for measure in song:
            chord, note_sequence = song[measure]['chord'], song[measure]['note_sequence']
            inputs.append([one_hot_encoding(note, root_list) for note in note_sequence])
            targets.append(one_hot_encoding(chord, chord_list))
        vectorized_songs.append([inputs, targets])
    return vectorized_songs


def one_hot_encoding(data, data_list):
    """Return the one hot vector."""
    vectors = [0] * len(data_list)
    index = data_list.index(data)
    vectors[index] = 1
    return vectors


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)

    train = preprocess(config['raw_train'])
    test = preprocess(config['raw_test'])

    if not os.path.exists(config['npy_train']) or not os.path.exists(config['npy_test']):
        np.save(config['npy_train'], train)
        np.save(config['npy_test'], test)
        print('save numpy file.')
    else:
        print('numpy file already exist.')