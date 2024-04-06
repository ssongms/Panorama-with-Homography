"""

* 두 이미지를 Stitch하는데 필요한 모듈입니다.


"""

import cv2
import numpy as np


# 네 개의 코너 좌표를 반환하는 함수입니다.
def getCorners(image):
    # 코너 좌표를 저장할 3차원 배열을 생성합니다.
    corners = np.zeros((4, 1, 2), dtype=np.float32)

    # 파라미터 이미지의 형태로부터 높이와 너비를 변수에 저장합니다.
    shape = image.shape
    y = shape[0]
    x = shape[1]

    # 이미지의 좌상단, 좌하단, 우하단, 우상단의 좌표를 갖습니다.
    corners[0] = [0, 0]
    corners[1] = [0, y]
    corners[2] = [x, y]
    corners[3] = [x, 0]

    # 최종결과를 반환합니다.
    return corners


# resultDemension에서 x축 또는 y축에서의 최솟값을 반환하는 함수입니다.
def getMin(resultDemension, axis):
    # 파라미터로 x축을 지정하면 x축에 대한 최솟값을 반환
    if axis == "x":
        x_min = int(resultDemension.min(axis=0).reshape(-1, order="A")[0])
        return x_min
    # 파라미터로 y축을 지정하면 y축에 대한 최솟값을 반환
    if axis == "y":
        y_min = int(resultDemension.min(axis=0).reshape(-1, order="A")[1])
        return y_min


# resultDemension에서 x축 또는 y축에서의 최댓을 반환하는 함수입니다.
def getMax(resultDemension, axis):
    # 파라미터로 x축을 지정하면 x축에 대한 최댓값을 반환
    if axis == "x":
        x_max = int(resultDemension.max(axis=0).reshape(-1, order="A")[0])
        return x_max
    # 파라미터로 y축을 지정하면 y축에 대한 최댓값을 반환
    if axis == "y":
        y_max = int(resultDemension.max(axis=0).reshape(-1, order="A")[1])
        return y_max


# 호모그래피 H를 기반으로 img2를 Img1에 stitch하는 함수입니다.(Panorama)
def stitch(img1, img2, H):
    # 각 이미지의 너비와 높이를 추출하여 변수에 저장합니다.
    width1, height1 = img1.shape[:2]
    width2, height2 = img2.shape[:2]

    # getCorners 함수를 사용하여 각 이미지의 네 꼭지점 좌표를 계산합니다.
    dimension1 = getCorners(img1).reshape(-1, 1, 2)
    dimension2_t = getCorners(img2).reshape(-1, 1, 2)

    # 호모그래피 매트릭스를 적용하여 원근 변환을 수행합니다.
    dimension2 = cv2.perspectiveTransform(dimension2_t, H)

    # 얻은 꼭지점 좌표들을 result_dimension 배열에 concatenate하여 하나의 배열로 만듭니다.
    result_dimension = np.concatenate((dimension1, dimension2))

    # 연결된 배열에서 최소 x, y와 최대 x, y를 구합니다.
    # 두 이미지를 입어붙인 후 전체 이미지의 크기를 결정하는데 사용됩니다.
    x_min = getMin(result_dimension, "x")
    y_min = getMin(result_dimension, "y")
    x_max = getMax(result_dimension, "x")
    y_max = getMax(result_dimension, "y")

    # 최소 x, y 좌표값을 이용하여 이미지를 이동시키기 위한 변환 매트릭스 H2를 생성합니다.
    b = [-x_min, -y_min]
    H2 = np.array([[1, 0, b[0]], [0, 1, b[1]], [0, 0, 1]])

    # 두 번째 이미지 (img2)를 첫 번째 이미지 (img1)에 맞게 원근 변환합니다.
    # 변환 매트릭스는 H2.dot(H)로 계산됩니다.
    # 결과 이미지의 크기는 이전에 계산한 최대, 최소 좌표값을 통해 결정됩니다.
    result_img = cv2.warpPerspective(
        img2,
        H2.dot(H),
        (x_max - x_min, y_max - y_min),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    #  첫 번째 이미지를 결과 이미지에 복사하고 위치를 조절하여 두 이미지를 이어붙입니다.
    result_img[
        -y_min : img1.shape[:2][0] + -y_min, -x_min : img1.shape[:2][1] + -x_min
    ] = img1

    # cv2 톤 매핑 함수에 적용하기 위해 형식을 변환합니다.
    stitched_img_float32 = result_img.astype(np.float32)

    # 톤 매핑 알고리즘을 생성한 뒤 톤 매핑을 적용합니다.
    # 파노라마의 이미지 결과가 더 자연스러워질 수 있습니다.
    tonemap = cv2.createTonemapReinhard()
    finalImg = tonemap.process(stitched_img_float32)

    # 결과 이미지를 반환합니다.
    return finalImg
