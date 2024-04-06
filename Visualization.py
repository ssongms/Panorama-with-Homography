"""

* 시각화 관련된 모듈입니다.
* 진행 과정을 눈으로 보고, 결과를 확인하기 위한 함수를 구현했습니다.

"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# 이미지와 키포인트를 기반으로 좌표를 마킹한 이미지를 반환하는 함수입니다.
def drawKeypoints(image, keypoints):
    # Actual parameter에 영향을 주지 않기 위해서 복사해서 사용합니다.
    markedImg = np.copy(image)

    # 키 포인트 좌표를 이미지에 표시합니다.
    # 반지름은 3, 두께가 3인 원으로, 색은 BGR(0,255,0)으로 표시합니다.
    for keypoint in keypoints:
        x, y = keypoint
        cv2.circle(markedImg, (x, y), 3, (0, 255, 0), 3)

    # 결과 이미지를 리턴합니다.
    return markedImg


# 두 이미지의 대응점을 라인으로 이은 뒤 시각화해서 보여주는 함수입니다.
def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
    # 새로운 figure과 서브 플롯 객체인 ax를 생성합니다.
    # 서브플롯 객체 ax을 통해 개별적인 플롯을 관리할 수 있습니다.
    figure, ax = plt.subplots(figsize=(30, 10))

    # 첫 번째 이미지와 두번째 이미지를 이어 붙이는 과정입니다.
    ax.imshow(image1)  # 첫 번째 이미지(image1)를 현재의 서브플롯에 표시합니다.
    ax.imshow(
        image2,  # 두 번째 이미지(image2)를 현재의 서브플롯에 표시합니다.
        extent=[image1.shape[1], image1.shape[1] + image2.shape[1], image2.shape[0], 0],
    )  # extent 설정을 통해 두 이미지가 나란하게 보여집니다.

    # 각 이미지에 keypoint를 marking합니다.
    # image1에는 파란색으로 Mark하고 image2에는 빨간색으로 Mark합니다.
    ax.scatter(keypoints1[:, 0], keypoints1[:, 1], c="b", s=6)
    ax.scatter(keypoints2[:, 0] + image1.shape[1], keypoints2[:, 1], c="r", s=6)

    # 매칭된 키포인트를 line으로 연결합니다.
    for match in matches:
        start_point = keypoints1[match[0]]
        end_point = keypoints2[match[1]] + np.array([image1.shape[1], 0])

        # 매칭된 line이 서로 구분이 될 수 있도록 line 색상을 다르게 하기 위한 난수입니다.
        randomColor = np.random.rand(3)

        # 라인 configuration 입니다.
        line = Line2D(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linewidth=2,
            color=randomColor,
        )
        ax.add_line(line)  # line을 추가합니다.

    # 결과 이미지를 시각화합니다.
    plt.show()
