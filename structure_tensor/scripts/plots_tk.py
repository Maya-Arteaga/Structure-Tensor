from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from matplotlib import pyplot as plt
import numpy as np


def plot_ST(image, results, path, size, n_rows, n_columns, c):

    img = np.array(image)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', extent=[0, np.shape(image)[1], 0, np.shape(image)[0]])
    plt.axis('off')

    # display everything:
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', extent=[0, np.shape(image)[1], 0, np.shape(image)[0]])
    plt.axis('off')

    mean = np.zeros((n_rows, n_columns))
    # trace = np.zeros((n_rows, n_columns))
    quo = np.zeros((n_rows, n_columns))
    FA = np.zeros((n_rows, n_columns))

    for i in range(1, n_rows - 1):
        for j in range(1, n_columns - 1):
            x = results[i, j]
            if x[1][0] == x[1][1]:
                pass
            else:
                # plot the ACG
                ax.plot(c[0] + size * j + x[0][0, :] * size,
                        c[1] + size * (n_rows - 1 - i) + x[0][1, :] * size,
                        lw=0.2, color='darkorange', ls='-')  # lw = 2

                # plot an arrow with the principal direction
                if x[1][0] < x[1][1]:
                    ax.arrow(c[0] + size * j, c[1] + size * (n_rows - 1 - i),
                             x[2][0, 0] * size / n_columns, x[2][1, 0] * size / n_columns,
                             width=1, head_width=2, ec='limegreen', fc='limegreen')

                    mean[i, j] = (x[1][0] + x[1][1]) / 2
                    # trace[i, j] = x[1][0] + x[1][1]
                    quo[i, j] = x[1][0] / x[1][1]
                    FA[i, j] = (x[1][1] - x[1][0]) / (2*(np.sqrt(x[1][1]**2 + x[1][0]**2)))

                if x[1][1] < x[1][0]:
                    ax.arrow(c[0] + size * j, c[1] + size * (n_rows - 1 - i),
                             x[2][0, 1] * size / n_columns, x[2][1, 1] * size / n_columns,
                             width=1, head_width=2, ec='limegreen', fc='limegreen')

                    mean[i, j] = (x[1][0] + x[1][1]) / 2
                    # trace[i, j] = x[1][0] + x[1][1]
                    quo[i, j] = x[1][1] / x[1][0]
                    FA[i, j] = (x[1][0] - x[1][1]) / (2*(np.sqrt(x[1][1]**2 + x[1][0]**2)))


                del x

    plt.savefig(str(path)+'_ST_'+str(size)+'.png', bbox_inches='tight', dpi=800)
    plt.close()

    im = plt.imshow(mean, cmap="viridis")
    plt.colorbar(im)
    plt.savefig(str(path)+'_mean_'+str(size)+'.png', bbox_inches='tight', dpi=800)
    plt.close()

    # im = plt.imshow(trace, cmap="viridis")
    # plt.colorbar(im)
    # plt.savefig(str(path)+'_trace_'+str(size)+'.png', bbox_inches='tight', dpi=800)
    # plt.close()

    im = plt.imshow(quo, cmap="plasma", vmin=0)
    plt.colorbar(im)
    plt.savefig(str(path) + '_disp_' + str(size) + '.png', bbox_inches='tight', dpi=800)
    plt.close()
    
    #print('que passa')
    im = plt.imshow(FA, cmap="plasma", vmin=0)
    plt.colorbar(im)
    plt.savefig(str(path) + '_FA_' + str(size) + '.png', bbox_inches='tight', dpi=800)
    plt.close()
    #print('eissss')

