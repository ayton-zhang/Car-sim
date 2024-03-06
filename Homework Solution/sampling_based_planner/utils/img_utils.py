import cv2
import numpy as np

def binary_to_grid(binary, grid_size):
    # 计算栅格行列数
    rows, cols = binary.shape[:2]
    grid_rows = int(rows / grid_size)
    grid_cols = int(cols / grid_size)

    print(rows)
    print(cols)
    # 创建二维列表
    grid = [[False for j in range(grid_cols)] for i in range(grid_rows)]

    # 遍历二值化图像的像素
    for i in range(rows):
        for j in range(cols):
            # 计算像素所在栅格的行列号
            row = int(i / grid_size)
            col = int(j / grid_size)

            # 如果像素为黑色，则将对应的栅格设置为True
            if binary[i, j] == 0:
                grid[grid_rows - row - 1][col] = True

    return grid

def discretize_image(image, grid_size):
    """将二值化图像离散化成栅格

    Args:
        image (numpy.ndarray): 二值化图像，每个像素值为0或255
        grid_size (int): 离散化后栅格的大小，即每个网格的边长

    Returns:
        list: 二维栅格地图，True表示黑色，False表示白色
    """
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算栅格的行数和列数
    num_rows = height // grid_size
    num_cols = width // grid_size

    # 创建一个二维列表来表示栅格地图
    grid_map = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    # 遍历栅格地图的每个网格
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前网格的左上角坐标和右下角坐标
            x1 = col * grid_size
            y1 = row * grid_size
            x2 = (col + 1) * grid_size
            y2 = (row + 1) * grid_size

            # 检查当前网格中是否有黑色像素
            grid_image = image[y1:y2, x1:x2]
            # print(grid_image)
            if np.any(grid_image == 0):
                grid_map[row][col] = 1

    return grid_map

# flip_img = cv2.flip(binary_img, 0)
    # rot_image = cv2.rotate(binary_img, cv2.ROTATE_90_CLOCKWISE)
    # grid_map = discretize_image(binary_img, 2)
    # grid_map = binary_to_grid(binary_img, grid_size)


     # 显示图像
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.imshow(binary_img, cmap='gray')
    # plt.show()