import os
from ffmpy import FFmpeg

dataset_path = '../FakeAvCeleb/RealVideo-FakeAudio' #RealAudio'
output_path = '../FakeAvCeleb/cc/fake'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("Path {0} created.".format(output_path))

for path, subdirs, files in os.walk(dataset_path):
    for name in files:
        if name.endswith('.mp4'):
            input_name = os.path.join(path, name)
            output_name = name[:-4] + '.wav'

            output_name = os.path.basename(os.path.normpath(path)) + '_' + output_name
            output_name = os.path.join(output_path, output_name)
            
            if not os.path.exists(output_name):
                ff = FFmpeg(
                    inputs={input_name: None}, 
                    outputs={output_name: '-af silenceremove=1:0:-40dB -to 00:00:02'}
                )
                ff.run()
