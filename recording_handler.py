import glob
import enum
import os.path, os

class Finger(enum.Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

class Hand(enum.Enum):
    LEFT = 0
    RIGHT = 1

class Vec:
    def __init__(self, x, y, z):
        self.data = [x, y, z]

    def __repr__(self):
        return str(self.data)

class FingerData:
    def __init__(self):
        # real number
        self.ang = None

        # vector
        self.dir = None
        self.lm_pos = None
        self.mov_dir = None
        self.pos = None
        self.rel_pos = None
        self.tip_pos = None

    def get_feature(self, feature_name_str):
        fns = feature_name_str

        if fns == 'ang':
            return self.ang
        elif fns == 'dir_x':
            return self.dir.data[0]
        elif fns == 'dir_y':
            return self.dir.data[1]
        elif fns == 'dir_z':
            return self.dir.data[2]
        elif fns == 'mov_dir_x':
            return self.mov_dir.data[0]
        elif fns == 'mov_dir_y':
            return self.mov_dir.data[1]
        elif fns == 'mov_dir_z':
            return self.mov_dir.data[2]
        elif fns == 'pos_x':
            return self.pos.data[0]
        elif fns == 'pos_y':
            return self.pos.data[1]
        elif fns == 'pos_z':
            return self.pos.data[2]
        elif fns == 'rel_pos_x':
            return self.rel_pos.data[0]
        elif fns == 'rel_pos_y':
            return self.rel_pos.data[1]
        elif fns == 'rel_pos_z':
            return self.rel_pos.data[2]
        elif fns == 'tip_pos_x':
            return self.tip_pos.data[0]
        elif fns == 'tip_pos_y':
            return self.tip_pos.data[1]
        elif fns == 'tip_pos_z':
            return self.tip_pos.data[2]
        elif fns == 'lm_pos_x':
            return self.lm_pos.data[0]
        elif fns == 'lm_pos_y':
            return self.lm_pos.data[1]
        elif fns == 'lm_pos_z':
            return self.lm_pos.data[2]
        else:
            assert False

    # def get_dict(self):
    #     finger_dict = {}

    #     finger_dict['ang'] = self.ang

    #     finger_dict['dir_x'] = [d[0] for d in self.dir.data]
    #     finger_dict['dir_y'] = [d[1] for d in self.dir.data]
    #     finger_dict['dir_z'] = [d[2] for d in self.dir.data]

    #     finger_dict['mov_dir_x'] = [d[0] for d in self.mov_dir.data]
    #     finger_dict['mov_dir_y'] = [d[1] for d in self.mov_dir.data]
    #     finger_dict['mov_dir_z'] = [d[2] for d in self.mov_dir.data]

    #     finger_dict['pos_x'] = [d[0] for d in self.pos.data]
    #     finger_dict['pos_y'] = [d[1] for d in self.pos.data]
    #     finger_dict['pos_z'] = [d[2] for d in self.pos.data]

    #     finger_dict['rel_pos_x'] = [d[0] for d in self.rel_pos.data]
    #     finger_dict['rel_pos_y'] = [d[1] for d in self.rel_pos.data]
    #     finger_dict['rel_pos_z'] = [d[2] for d in self.rel_pos.data]

    #     finger_dict['tip_pos_x'] = [d[0] for d in self.tip_pos.data]
    #     finger_dict['tip_pos_y'] = [d[1] for d in self.tip_pos.data]
    #     finger_dict['tip_pos_z'] = [d[2] for d in self.tip_pos.data]

    #     finger_dict['lm_pos_x'] = [d[0] for d in self.lm_pos.data]
    #     finger_dict['lm_pos_y'] = [d[1] for d in self.lm_pos.data]
    #     finger_dict['lm_pos_z'] = [d[2] for d in self.lm_pos.data]

    #     return finger_dict



    @staticmethod
    def parse_finger(finger_dict, index):
        finger = FingerData()

        finger.ang = finger_dict['ang'][index]

        finger.dir = Vec(finger_dict['dir_x'][index], finger_dict['dir_y'][index], finger_dict['dir_z'][index])
        finger.mov_dir = Vec(finger_dict['mov_dir_x'][index], finger_dict['mov_dir_y'][index], finger_dict['mov_dir_z'][index])
        finger.pos = Vec(finger_dict['pos_x'][index], finger_dict['pos_y'][index], finger_dict['pos_z'][index])
        finger.rel_pos = Vec(finger_dict['rel_pos_x'][index], finger_dict['rel_pos_y'][index], finger_dict['rel_pos_z'][index])
        finger.tip_pos = Vec(finger_dict['tip_pos_x'][index], finger_dict['tip_pos_y'][index], finger_dict['tip_pos_z'][index])
        finger.lm_pos = Vec(finger_dict['lm_pos_x'][index], finger_dict['lm_pos_y'][index], finger_dict['lm_pos_z'][index])

        return finger

class HandData:
    def __init__(self):
        #self.valid = None

        self.thumb = None
        self.index = None
        self.middle = None
        self.ring = None
        self.pinky = None

    def get_finger(self, finger_name_str):
        fns = finger_name_str

        if fns == 'thumb':
            return self.thumb
        elif fns == 'index':
            return self.index
        elif fns == 'middle':
            return self.middle
        elif fns == 'ring':
            return self.ring
        elif fns == 'pinky':
            return self.pinky
        else:
            assert False

    
    # def get_dict(self):
    #     hand_dict = {}

    #     hand_dict['thumb'] = self.thumb.get_dict()
    #     hand_dict['index'] = self.index.get_dict()
    #     hand_dict['middle'] = self.middle.get_dict()
    #     hand_dict['ring'] = self.ring.get_dict()
    #     hand_dict['pinky'] = self.pinky.get_dict()

    #     return hand_dict
    
    @staticmethod
    def parse_hand(hand_dict, index):
        hand = HandData()

        hand.thumb = FingerData.parse_finger(hand_dict['thumb'], index)
        hand.index = FingerData.parse_finger(hand_dict['index'], index)
        hand.middle = FingerData.parse_finger(hand_dict['middle'], index)
        hand.ring = FingerData.parse_finger(hand_dict['ring'], index)
        hand.pinky = FingerData.parse_finger(hand_dict['pinky'], index)

        #hand.valid = hand_dict['thumb']['valid'] == 1

        return hand

class UnitData:
    def __init__(self):
        self.timestamp = None
        
        self.left_hand = None
        self.right_hand = None

        self.is_keytap = False
        self.tapping_hand = None
        self.key_name = None

    def get_hand(self, hand_name_str):
        if hand_name_str == 'left':
            return self.left_hand
        elif hand_name_str == 'right':
            return self.right_hand
        else:
            assert False

# path, left/right, finger name, feature name
# example: "C:/blabla/data/", "left", "thumb", "pos_x"
def get_full_filepath(path, which_hand, which_finger, which_feature):
    suffix = which_hand.lower() + "_" + which_finger.upper() + "_" + which_feature.upper() + ".txt"
    return os.path.join(path, suffix)

def read_data_from_tab_delimited_file(path):
    with open(path) as f:
        line = f.readline()
        line = line.replace("\n", "").split("\t")
        return line

def parse_vectors(file_x, file_y, file_z):
    x = [float(d) for d in read_data_from_tab_delimited_file(file_x)]
    y = [float(d) for d in read_data_from_tab_delimited_file(file_y)]
    z = [float(d) for d in read_data_from_tab_delimited_file(file_z)]

    return list(zip(x, y, z))

class Label:
    def __init__(self):
        self.which_hand = None
        self.index = None

    @staticmethod
    def parse_label(label_string):
        label = Label()

        label_string = label_string.replace('(', '').replace(')', '')
        hand_name, index = label_string.split(',')
        
        label.which_hand = Hand.LEFT if hand_name == 'left' else Hand.RIGHT
        label.index = int(index)

        return label

def get_data_dict(path):
    hand_names = "left", "right"
    finger_names = "thumb", "index", "middle", "ring", "pinky"
    feature_names = "ang", "dir_x", "dir_y", "dir_z", "mov_dir_x", "mov_dir_y", "mov_dir_z", "pos_x", "pos_y", "pos_z", "rel_pos_x", "rel_pos_y", "rel_pos_z", "tip_pos_x", "tip_pos_y", "tip_pos_z", "lm_pos_x", "lm_pos_y", "lm_pos_z" #, "valid"

    # fill data_dict such that data[hand][finger][feature] would give a list of floats
    data_dict = {}

    # read all data
    for hand in hand_names:
        data_dict[hand] = {}

        for finger in finger_names:
            data_dict[hand][finger] = {}

            for feature in feature_names:
                filepath = get_full_filepath(path, hand, finger, feature)
                data_string_list = read_data_from_tab_delimited_file(filepath)
                data_float_list = list(map(float, data_string_list))

                data_dict[hand][finger][feature] = data_float_list
    return data_dict

class RecordingHandler:
    @staticmethod
    def parse(folder_path):
        # Read timestamps, keys, labels
        timestamp_path = os.path.join(folder_path, 'timestamps.txt')
        keys_path = os.path.join(folder_path, 'keys.txt')
        labels_path = os.path.join(folder_path, 'labels.txt')

        timestamps = [int(d) for d in read_data_from_tab_delimited_file(timestamp_path)[:-1]]
        keys = read_data_from_tab_delimited_file(keys_path)[:-1]
        labels = read_data_from_tab_delimited_file(labels_path)[:-1]

        # Parse labels
        labels = [ Label.parse_label(x) for x in labels ]

        data_dict = get_data_dict(folder_path)

        label_index = 0

        # Initially, no unit data -- to be filled
        unit_datas = []

        for i, t in enumerate(timestamps):
            unit_data = UnitData()

            # Timestamp
            unit_data.timestamp = t

            # Hands
            unit_data.left_hand = HandData.parse_hand(data_dict['left'], i)
            unit_data.right_hand = HandData.parse_hand(data_dict['right'], i)

            # Is that index included in the labels as the next label? Check for key tap
            if label_index < len(labels) and labels[label_index].index == i:
                unit_data.is_keytap = True
                unit_data.tapping_hand = labels[label_index].which_hand
                unit_data.key_name = keys[label_index]
                label_index += 1

            unit_datas.append(unit_data)

        return unit_datas
    
    @staticmethod
    def get_slice(unit_datas, starting_timestamp, ending_timestamp):
        pass

    @staticmethod
    def dump(unit_datas, folder_path):
        # TODO: Do a quick check if timestamps are consistent
        if len(unit_datas) == 0:
            print("Nothing to dump.")
            return False

        ### Folder handling
        # Check if dir exists
        if os.path.exists(folder_path):
            if os.path.isdir(folder_path):
                # directory
                # check if empty
                if os.listdir(folder_path):
                    # not empty
                    print("There exists files at folder:", folder_path)
                    assert False
            else:
                # file
                print("There exists a file at:", folder_path)
                assert False
        else:
            # create folder
            os.mkdir(folder_path)
        
        # Good to go with the directory

        # Recover timestamps
        timestamps = [u.timestamp for u in unit_datas]
        timestamps_string = '\t'.join([str(t) for t in timestamps]) + '\t'

        # Recover labels and keys
        labels_string = ''
        keys_string = ''

        for i, u in enumerate(unit_datas):
            if u.is_keytap:
                label_str = '(' + ('left' if u.tapping_hand == Hand.LEFT else 'right') + ',' + str(i) + ')\t'
                labels_string += label_str
                keys_string += u.key_name + '\t'


        # Write timestamps.txt, keys.txt, and labels.txt
        with open( os.path.join(folder_path, 'timestamps.txt'), 'w+' ) as f:
            f.write(timestamps_string + '\n')
        with open( os.path.join(folder_path, 'keys.txt'), 'w+'  ) as f:
            f.write(keys_string + '\n')
        with open( os.path.join(folder_path, 'labels.txt'), 'w+'  ) as f:
            f.write(labels_string + '\n')

        # Recover data dict -- just in case
        data_dict = {}
        
        hand_names = "left", "right"
        finger_names = "thumb", "index", "middle", "ring", "pinky"
        feature_names = "ang", "dir_x", "dir_y", "dir_z", "mov_dir_x", "mov_dir_y", "mov_dir_z", "pos_x", "pos_y", "pos_z", "rel_pos_x", "rel_pos_y", "rel_pos_z", "tip_pos_x", "tip_pos_y", "tip_pos_z", "lm_pos_x", "lm_pos_y", "lm_pos_z" #, "valid"

        # Initialize dict
        for hand_name in hand_names:
            data_dict[hand_name] = {}

            for finger_name in finger_names:
                data_dict[hand_name][finger_name] = {}

                for feature_name in feature_names:

                    dat = []
                    for u in unit_datas:
                        single_dat = u.get_hand(hand_name).get_finger(finger_name).get_feature(feature_name)
                        if single_dat == 0.0:
                            single_dat = 0
                        dat.append(single_dat)

                    # in case it is needed in the future
                    data_dict[hand_name][finger_name][feature_name] = dat
                    
                    feature_list = data_dict[hand_name][finger_name][feature_name]
                    feature_str_list = [ str(f) for f in feature_list ]
                    feature_str = '\t'.join(feature_str_list) + '\n'

                    filepath = get_full_filepath(folder_path, hand_name, finger_name, feature_name)

                    with open(filepath, 'w+') as f:
                        f.write(feature_str)

        return True
    
    @staticmethod
    def eliminate_same_timestamped(unit_datas):
        # Best effort to keep all key taps in the list
        result = []

        last_ts = 0
        to_add = None # check if any data with key tap has the same timestamp with some other data. so that, do not miss any key tap data
        keytap_found = False # associated with the last timestamp

        for u in unit_datas:
            if u.timestamp == last_ts:
                # among the ones with the same timestamp
                if not keytap_found and u.is_keytap:
                    to_add = u
                    keytap_found = True
                    result.append(u)
                elif keytap_found and u.is_keytap:
                    print('Warning at eliminate_same_timestamped(): Multiple keytaps at the same timestamp')
                elif keytap_found and not u.is_keytap:
                    # do nothing
                    continue
                elif not keytap_found and not u.is_keytap:
                    # just keep the most recent
                    to_add = u

            else:
                last_ts = u.timestamp

                if not keytap_found and to_add != None:
                    result.append(to_add)
                
                to_add = u

                if u.is_keytap:    
                    keytap_found = True
                    result.append(u)
                else:
                    keytap_found = False
        
        return result
    
    @staticmethod
    def count_keytaps(unit_datas):
        return sum( [1 if u.is_keytap else 0 for u in unit_datas] )
            



#recording_folder = 'C:/Users/necip/Desktop/inference/keytap-detection/data/EO-frequented/'

#unit_datas = RecordingHandler.parse(recording_folder)

#new_unit_datas = RecordingHandler.eliminate_same_timestamped(unit_datas)

#RecordingHandler.dump(new_unit_datas, 'repetition-eliminated-eo-freq/')

# predicted= [542, 615, 649, 682, 744, 790, 840, 874, 908, 1000, 1128, 1310, 1343, 1435, 1484, 1544, 1586, 1647, 1687, 1728, 1792, 1835, 1878, 1926, 2052, 2114, 2171, 2201, 2247, 2317, 2386, 2430, 2460, 2546, 2592, 2733, 2781, 2871, 2941, 2971, 3026, 3063, 3230, 3302, 3389, 3419, 3463, 3521, 3571, 3624, 3675, 3726, 3780, 3888]      
# gt= [547, 618, 649, 682, 699, 744, 791, 808, 840, 874, 909, 1001, 1128, 1310, 1350, 1443, 1484, 1544, 1585, 1647, 1687, 1728, 1792, 1835, 1878, 1901, 1923, 2052, 2114, 2203, 2247, 2268, 2337, 2384, 2430, 2455, 2546, 2594, 2733, 2781, 2871, 2941, 2965, 3037, 3062, 3086, 3229, 3302, 3388, 3415, 3466, 3519, 3571, 3624, 3675, 3726, 3780, 3888]

# Afsah Random5
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Afsah/Random5/'
# start_index = 5261
# part1_ends = 8302
# part2_ends = 11425

# Afsah Random10
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Afsah/Random10/'
# start_index = 3413
# part1_ends = 4519
# part1_ends = 9087
# part2_ends = 14537
# unit_datas = RecordingHandler.parse(recording_folder)

# Afsah Random15
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Afsah/Random15/'
# start_index = 4149
# part1_ends = 13081
# part2_ends = 21241
# unit_datas = RecordingHandler.parse(recording_folder)


# Ahmed Random5
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Ahmed/Random5/'
# start_index = 4066
# part1_ends = 6832
# part2_ends = 9682
# unit_datas = RecordingHandler.parse(recording_folder)

# Ahmed Random10
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Ahmed/Random10/'
# start_index = 4377
# part1_ends = 5677
# old part1_ends = 9458
# old part2_ends = 17200
# unit_datas = RecordingHandler.parse(recording_folder)

# Ahmed Random15
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Ahmed/Random15/'
# start_index = 5365
# part1_ends = 15987
# part2_ends = 24400
# unit_datas = RecordingHandler.parse(recording_folder)

# Jinchun Random5
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Jinchun/Random5/'
# start_index = 5019
# part1_ends = 7129
# part2_ends = 9490
# unit_datas = RecordingHandler.parse(recording_folder)

# Jinchun Random10
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Jinchun/Random10/'
# start_index = 4633
# part1_ends = 5423
# old part1_ends = 8485
# old part2_ends = 12323
# unit_datas = RecordingHandler.parse(recording_folder)

# Jinchun Random15
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Jinchun/Random15/'
# start_index = 4159
# part1_ends = 11470
# part2_ends = 18181
# unit_datas = RecordingHandler.parse(recording_folder)




# Jinchun Mail
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Jinchun/Mail/'
# start_index = 6112
# part1_ends = 10926
# part2_ends = 16523
# part3_ends = 21328
# part4_ends = 26584
# part5_ends = 33856
# unit_datas = RecordingHandler.parse(recording_folder)

# Afsah Mail
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Afsah/Mail/'
# start_index = 7494
# part1_ends = 14772
# part2_ends = 21919
# part3_ends = 28444
# part4_ends = 34879
# part5_ends = 42348
# unit_datas = RecordingHandler.parse(recording_folder)

# Ahmed Mail
# recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Ahmed/Mail/'
# start_index = 3820
# part1_ends = 9752
# part2_ends = 15552
# part3_ends = 19883
# part4_ends = 24529
# part5_ends = 29841
# unit_datas = RecordingHandler.parse(recording_folder)

recording_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Ahmed/Random10/part3'
dump_folder = 'C:/Users/necip/Desktop/AR_Recordings/AR-Keyboard-Experiments/UserSide/Ahmed/Random10/'
unit_datas = RecordingHandler.parse(recording_folder)

sub_datas = []

# Fill keytap moments
keytap_moments = []

for i, u in enumerate(unit_datas):
    if u.is_keytap:
        if u.key_name == '_BS_':
            keytap_moments.pop()
        else:
            keytap_moments.append(i)

keytap_moments[0] = 0
keytap_moments[-1] = len(unit_datas) - 1

# Get chunk start and end indices
chunk_indices = []
got_data = 0
start_ind = 0

data_per_chunk = 10

for i, k in enumerate(keytap_moments):    
    got_data += 1

    if got_data == data_per_chunk:
        got_data = 0

        if len(keytap_moments) == (i+1):
            end_ind = keytap_moments[-1]
        else:
            end_ind = (keytap_moments[i] + keytap_moments[i+1]) // 2
        
        chunk_indices.append( (start_ind, end_ind) )

        start_ind = end_ind


# filled chunk indices -- now dump the data
for i, (start_ind, end_ind) in enumerate(chunk_indices):
    chunk = unit_datas[ start_ind:end_ind ]

    chunk_folder_p = os.path.join(dump_folder, 'p-{}'.format(i+1))

    RecordingHandler.dump(chunk, chunk_folder_p)

exit()







for i in range(len(unit_datas)-1, -1, -1 ) in enumerate(unit_datas):
    if len(sub_datas) == 5:
        # dump and clear
        RecordingHandler.dump(sub_datas, os.path.join(recording_folder, 'p-{}'.format(next_dump_id)))
        sub_datas = []
    

    


keytap_count = 0
for i, u in enumerate(unit_datas):
    if u.is_keytap:
        if u.key_name == '_BS_':
            keytap_count -= 2

        if (not done_start) and keytap_count == 10:
            print('')
            keytap_count = 0
            done_start = True

        print( u.key_name + ' @ ' + str(i) )
        if (keytap_count+1) % 5 == 0:
            print('')

        keytap_count += 1

exit()

for i, u in enumerate(unit_datas):
    if i >= start_index:
        if i <= part1_ends:
            part1.append(u)
        
        # if i <= part2_ends:
        #     part2.append(u)

        # if i <= part3_ends:
        #     part3.append(u)

        # if i <= part4_ends:
        #     part4.append(u)

        # if i <= part5_ends:
        #     part5.append(u)

        

        #part3.append(u)




# for i, u in enumerate(unit_datas):
#     if i >= start_index:
#         if i <= part1_ends:
#             part1.append(u)
        
#         # if i <= part2_ends:
#         #     part2.append(u)

#         # if i <= part3_ends:
#         #     part3.append(u)

#         # if i <= part4_ends:
#         #     part4.append(u)

#         # if i <= part5_ends:
#         #     part5.append(u)

        

#         #part3.append(u)



# RecordingHandler.dump(part1, os.path.join(recording_folder, 'part-10char'))
#RecordingHandler.dump(part1, os.path.join(recording_folder, 'part1'))
#RecordingHandler.dump(part2, os.path.join(recording_folder, 'part2'))
#RecordingHandler.dump(part3, os.path.join(recording_folder, 'part3'))
#RecordingHandler.dump(part4, os.path.join(recording_folder, 'part4'))
#RecordingHandler.dump(part5, os.path.join(recording_folder, 'part5'))










exit()
recording_folder = 'C:/Users/necip/Desktop/inference/keytap-detection/repetition-eliminated-eo-freq/'
unit_datas = RecordingHandler.parse(recording_folder)

ts_diff = []
for i in range( len(unit_datas) - 1 ):
    ts_diff.append(unit_datas[i+1].timestamp - unit_datas[i].timestamp)

for i, diff in enumerate(ts_diff):
    print("{}: {}".format(i, diff))

exit()



parent_folders_to_visit = \
    [ '../data/records-012320/left/test/*', \
      '../data/records-012320/left/train/*', \
      '../data/records-012320/right/test/*', \
      '../data/records-012320/right/train/*' ]

folders_to_visit = []
for pf in parent_folders_to_visit:
    folders_to_visit.extend(glob.glob(pf))

new_folder_names = []
for f in folders_to_visit:
    new_folder_names.append( os.path.join( f, "subsampled/" ) )

for oldp, newp in zip(folders_to_visit, new_folder_names):
    unit_datas = RecordingHandler.parse(oldp)

    new_unit_datas = RecordingHandler.eliminate_same_timestamped(unit_datas)

    RecordingHandler.dump(new_unit_datas, newp)
