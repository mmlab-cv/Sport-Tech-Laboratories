import c3d
import pandas as pd
import numpy as np


# preview what is in the file a bit
# with open("/home/giuliamartinelli/Documents/SportTech/MoCap File/Prova1.c3d",'rb') as hf:
#     reader = c3d.Reader(hf)
#     print('Frames:', len(list(reader.read_frames())))
#     for i, points, analog in reader.read_frames():
#         print('frame {}: point {}, analog {}'.format(
#             i, points.shape, analog.shape))
#         if i>5:
#             break


with open("/home/giuliamartinelli/Documents/SportTech/MoCap File/Prova1.c3d",'rb') as hf:
    all_fields = []
    reader = c3d.Reader(hf)
    scale_xyz = np.abs(reader.point_scale) # don't flip everything
    for frame_no, points, _ in reader.read_frames(copy=False):
        for (x, y, z, err, cam), label in zip(points, 
                                     reader.point_labels):
            if 'Sebastiano' in label.strip():
                c_field = {'frame': frame_no, 
                        'time': frame_no / reader.point_rate,
                        'point_label': label.strip()}
                c_field['x'] = scale_xyz*x
                c_field['y'] = scale_xyz*y
                c_field['z'] = scale_xyz*z
                c_field['err'] = err<0
                c_field['cam'] = cam<0
                all_fields += [c_field]
all_df = pd.DataFrame(all_fields)[['time', 'point_label', 'x', 'y', 'z']]

print(all_df.sample(5))