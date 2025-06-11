import rasterio
import numpy as np


def process_tiff_to_rgb(tif_file):
    """
    处理TIFF文件，提取RGB图像

    参数:
    tif_file: TIFF文件路径

    返回:
    rgb_image: RGB图像数组 (height, width, 3)
    """
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段（假设波段顺序为B02, B03, B04, B08, B12）
        bands = src.read()  # 形状为 (波段数, 高度, 宽度)，这里是 (5, height, width)

        # 分配波段（假设TIFF中的波段顺序为B02, B03, B04, B08, B12）
        blue = bands[0].astype(float)  # B02 - 蓝
        green = bands[1].astype(float)  # B03 - 绿
        red = bands[2].astype(float)  # B04 - 红
        nir = bands[3].astype(float)  # B08 - 近红外
        swir = bands[4].astype(float)  # B12 - 短波红外

        # 将图像数据压缩到0-255范围
        def normalize_band(band):
            band_min, band_max = band.min(), band.max()
            if band_max == band_min:  # 避免除零错误
                return np.zeros_like(band, dtype=np.uint8)
            return ((band - band_min) / (band_max - band_min)) * 255

        # 对每个波段进行归一化
        blue_normalized = normalize_band(blue).astype(np.uint8)
        green_normalized = normalize_band(green).astype(np.uint8)
        red_normalized = normalize_band(red).astype(np.uint8)

        # 创建 RGB 图像
        rgb_image = np.dstack((red_normalized, green_normalized, blue_normalized))

        return rgb_image


# 使用示例
if __name__ == "__main__":
    # 替换为你的TIFF文件路径
    tif_file_path = "your_file.tif"
    try:
        rgb_result = process_tiff_to_rgb(tif_file_path)
        print(f"RGB图像形状: {rgb_result.shape}")
        print("处理完成！")
    except Exception as e:
        print(f"处理出错: {e}")