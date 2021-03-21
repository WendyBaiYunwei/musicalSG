import glob
import csv
import os
import ntpath
import numpy as np


def one_hot_encoding(length, one_index):
    """Return the one hot vector."""
    vectors = [0] * length
    vectors[one_index] = 1
    return vectors


def make_test_npys(file_name, song_sequence):
    """Create npy file for each song in the test set."""
    file_path = "dataset/test_npy"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    np.save('%s/%s.npy' % (file_path, file_name.split('.')[0]), np.array(song_sequence))


def main():
    np.set_printoptions(threshold=np.inf)
    note_list = ['Cb', 'C0', 'Db', 'D0', 'E0', 'Eb', 'Fb', 'F0', 'Gb', 'G0', 'Ab', 'A0', 'Bb', 'B0', 'rest']
    note_dict = {}
    for i, note in enumerate(note_list):
        note_dict[note] = i 
    chord_dict = {}
    chord_list = ['Cb:maj', 'Cb:min',
                        'C0:maj', 'C0:min',
                        'Db:maj', 'Db:min',
                        'D0:maj', 'D0:min',
                        'E0:maj', 'E0:min',
                        'Eb:maj', 'Eb:min',
                        'Fb:maj', 'Fb:min',
                        'F0:maj', 'F0:min',
                        'Gb:maj', 'Gb:min',
                        'G0:maj', 'G0:min',
                        'Ab:maj', 'Ab:min',
                        'A0:maj', 'A0:min',
                        'B0:maj', 'B0:min',
                        'Bb:maj', 'Bb:min']

    for i, chord in enumerate(chord_list):
        chord_dict[chord] = i 

    print("1. Train set\n2. Test set")
    _input = input('Choose dataset to make npy file :')
    if _input == '1':
        file_path = 'dataset/csv_train/*.csv'
    elif _input == '2':
        file_path = 'dataset/csv_test/*.csv'
    else:
        print("input error")
        return None

    csv_files = glob.glob(file_path)
    note_dict_len = len(note_dict)
    chord_dict_len = len(chord_dict)

    # list for final input/target vector
    result_input_matrix = []
    result_target_matrix = []

    # make the matrix from csv data
    for csv_path in csv_files:
        csv_ins = open(csv_path, 'r', encoding='utf-8')
        reader = csv.reader(csv_ins)
        k = 0
        note_sequence = []
        song_sequence = []  # list for each song(each npy file) in the test set
        pre_measure = None
        for line in reader:
            k += 1
            if k == 1:
                continue
            if line[1] == 'X1':
                continue
            measure = int(line[1])
            chord_main = line[4].replace('#', '0')
            chord_type = line[5]
            chord = chord_main + ':' + 'min'
            if chord_type == "major" or chord_type == "dominant":
                chord = chord_main + ':' + 'maj'
            note = line[6].replace('#', '0')
            if note not in note_list or chord not in chord_list:
                continue
            # find one hot index
            
            chord_index = chord_dict[chord]
            note_index = note_dict[note]

            one_hot_note_vec = one_hot_encoding(note_dict_len, note_index)
            one_hot_chord_vec = one_hot_encoding(chord_dict_len, chord_index)

            if pre_measure is None:  # case : first line
                note_sequence.append(one_hot_note_vec)
                result_target_matrix.append(one_hot_chord_vec)

            elif pre_measure == measure:  # case : same measure note
                note_sequence.append(one_hot_note_vec)

            else:  # case : next measure note
                song_sequence.append(note_sequence)
                result_input_matrix.append(note_sequence)
                note_sequence = [one_hot_note_vec]
                result_target_matrix.append(one_hot_chord_vec)
            pre_measure = measure
        result_input_matrix.append(note_sequence)  # case : last measure note

        if _input == '2':
            # make npy file for each song
            make_test_npys(ntpath.basename(csv_path), song_sequence)

    if _input == '1':
        np.save('dataset/input_vector.npy', np.array(result_input_matrix))
        np.save('dataset/target_vector.npy', np.array(result_target_matrix))
        
    elif _input == '2':
        np.save('dataset/test_vector.npy', np.array(result_input_matrix))

if __name__ == '__main__':
    main()