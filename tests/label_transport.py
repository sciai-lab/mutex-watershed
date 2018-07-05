import mutex_watershed as mws
import numpy as np

height = 4

ws = mws.MutexWatershed(np.array([height,10]), np.array([[-1,0],[0,-1], [-3,0],[0,-3]]), 2, 1)
labels = np.ones((height,10))
labels[:, :5] = 2

ws.set_gt_labels(labels)
mask = np.zeros(labels.shape, dtype=np.bool)
mask[:, 4:6] = 1
ws.set_masked_gt_image(mask)

sorted_w = np.argsort(np.random.rand(height*10*4))

s = 1
while not ws.is_finised():
    ws.repulsive_ucc_mst_cut(sorted_w, s)
    print(ws.get_flat_c_label_image().reshape((height, 10)), ws.get_flat_region_gt_label_image().reshape((height, 10)))
    # print(ws.get_flat_applied_c_actions().reshape((4, height, 10))[[1,3]])
    print("")
    s += 1

