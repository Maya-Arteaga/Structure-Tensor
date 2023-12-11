from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
# import multiprocessing
import ST
import plots_tk


#path = "/Users/juanpablomayaarteaga/Desktop/structure_tensor/Data"
path= "/Users/juanpablomayaarteaga/Desktop/structure_tensor/EtOH/PCM/RID-43_EtOH_M/"
name = 'RID-43_EtOH_M_ROI'


image = Image.open(str(path)+str(name)+'.tif').convert('L')
img = np.array(image)


# size of squares where ST will be calculated:
size = 20
# parameters of the gaussian function:
sigma, rho = 2.3, 5
# border to take into account (4 is the truncate parameter of the gaussian filter by default)
k = 4

# number of pixels
p_rows, p_columns = np.shape(image)

# number of squares
n_rows = np.math.floor((p_rows - 2 * k) / size)
n_columns = np.math.floor((p_columns - 2 * k) / size)

# center point of each sub-image
c = [size / 2 + k, size / 2 + k]
# print(n_rows, n_columns)

# computing the ST of each image piece with multiple processes:
# pool = multiprocessing.Pool(processes=10)
results = {}

for i in range(0, n_rows):
    for j in range(0, n_columns):
        # C[i - 1, j - 1] = imgArray[i * size:(i + 1) * size + 2 * k, j * size:(j + 1) * size + 2 * k]
        piece = img[i * size:(i + 1) * size + 2 * k, j * size:(j + 1) * size + 2 * k]
        # indices = (sigma, rho, k, piece)
        # results[i-1, j-1] = pool.apply_async(ST.ST, args=(sigma, rho, k, C[i-1, j-1],))
        # results[i, j] = pool.starmap(ST.ST, indices)
        results[i, j] = ST.ST(sigma, rho, k, piece)
# pool.close()


# output = str(path)+str(name)+'_ST.npy'
# np.save(output, results)


plots_tk.plot_ST(img, results, str(path)+str(name), size, n_rows, n_columns, c)

