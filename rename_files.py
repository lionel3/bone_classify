import os

def get_all_files(root_dir, file_paths, file_names):
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
        else:
            get_all_files(path, file_paths, file_names)

root_dir = '/home/lionel/Desktop/bone_jpg/positive_ori'
file_path_names=[]
file_names=[]

get_all_files(root_dir, file_path_names, file_names)

for i in range(0, len(file_names)):
    file_path_name = root_dir + '/' + str(i).zfill(6) + '.jpg'
    os.rename(file_path_names[i], file_path_name);
