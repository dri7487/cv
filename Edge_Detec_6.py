## 최적화 분활 ##   
import skimage
import numpy as np
import cv2
import time

import skimage.future
import skimage.segmentation

img=skimage.data.coffee()
start=time.time()

slic=skimage.segmentation.slic(img,compactness=20,n_segments=600, start_label=1)
g = skimage.graph.rag_mean_color(img, slic, mode='similarity')
ncut=skimage.graph.cut_normalized(slic,g)
print(img.shape, 'image를 분할 하는데', time.time()-start, '초 소요')

marking = skimage.segmentation.mark_boundaries(img,ncut)
ncut_coffee = np.uint8(marking*255.0)

cv2.imshow('Normalized cut', cv2.cvtColor(ncut_coffee, cv2.COLOR_BGR2RGB))

cv2.waitKey()
cv2.destroyAllWindows()




