import os

dataset_path = '../JSUT' #../FakeAvCeleb/converted'
real_path = '../JSUT/cropped_genuine' #FakeAvCeleb/converted/real' #cc/real/'
fake_path = '../JSUT/cropped_multiband' #../FakeAvCeleb/converted/fake' #cc/fake'
subdirs = {'training', 'testing', 'validation' }

def create_subdirs(): 
    for name in subdirs:
        subdir_path = os.path.join(dataset_path, name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            tmp_dir = os.path.join(subdir_path, "real");
            os.makedirs(tmp_dir);
            tmp_dir = os.path.join(subdir_path, "fake");
            os.makedirs(tmp_dir);
            print('Directory {0} and subdirectories REAL and FAKE created.'.format(subdir_path))


def split_dataset(source_folder, real):
    count = 0
    tmp_dir = ""
    if (real == True):
        tmp_dir = "real"
    else:
        tmp_dir = "fake"
    output_path = dataset_path + '/fail'
    for filename in os.listdir(source_folder):
        if filename.endswith('.wav'):
            input_path = os.path.join(source_folder, filename)
            if (count % 10 == 4):
                output_path = os.path.join(dataset_path, 'testing', tmp_dir, filename)
            elif (count % 10 == 1 or count % 10 == 7):
                output_path = os.path.join(dataset_path, 'validation', tmp_dir, filename)
            else:
                output_path = os.path.join(dataset_path, 'training', tmp_dir, filename)
            os.rename(input_path, output_path)
            count = count + 1


create_subdirs()
split_dataset(real_path, True)
split_dataset(fake_path, False)

