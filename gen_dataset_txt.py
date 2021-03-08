import os
import glob

dataset_path = 'datasets/gta/'

def main():

    output_info = []

    road_count, sidewalk_count = 0,0

    for split_file in glob.glob('viper_*.txt'):
        _, split_set, split_type = split_file[:-4].split('_')
        print(split_set, split_type)
        with open(split_file,'r') as f:
            for track_info_str in f.readlines():
                track_info = track_info_str.strip().split(':')
                track_name = track_info[0]
                track_param_str = f"{split_set}/img/{track_name}/{track_name}_"
                if len(track_info) == 1: # use whole folder
                    start = 10
                    stop = max(int(f.split('/')[-1].split('.')[0].split('_')[1])
                    for f in glob.glob(dataset_path+track_param_str+"*.jpg"))
                else:
                    start, stop = map(int,track_info[1].split('-'))
                
                if split_type == 'sidewalk':
                    ranges = range(start,stop+1)
                else:
                    ranges = range(start,stop+1,10)
                for n in ranges:
                    color_fname = f"{dataset_path}{track_param_str}{n:05d}.jpg"
                    label_fname = color_fname.replace("img","cls")
                    label_fname = label_fname.replace("jpg","png")
                    if os.path.isfile(color_fname) and os.path.isfile(label_fname):
                        pair_info = ','.join([color_fname,label_fname])
                        output_info.append(pair_info)
                        if split_type == 'sidewalk':
                            sidewalk_count += 1
                        else:
                            road_count += 1
    with open('datasets/gta/gta5.txt','w+') as f:
        for output_instance in output_info:
            f.write(output_instance + "\n")
    print(road_count, sidewalk_count)

if __name__ == "__main__":
    main()