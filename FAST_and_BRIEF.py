"""

* FAST 알고리즘과 BRIEF 알고리즘 구현에 관련된 모듈입니다.
* 이미지에서 feature point를 찾고 descriptor를 구합니다. 
* 성능 향상을 위해 가우시안 스무딩(전처리) 또한 별도의 함수로 구현하고 적용했습니다.

"""

import numpy as np
from scipy.signal import convolve2d


# 가우시안 필터를 반환하는 함수입니다. 파라미터로 커널 사이즈와 sigma를 받습니다.
def getGaussianFilter(filter_size, sigma):
    #  가우시안 필터의 값을 저장할 배열을 초기화합니다. (2D array입니다.)
    gaussianFilter = [[0] * filter_size for _ in range(filter_size)]

    # 자연 상수를 정의합니다.
    exp = 2.71828
    pi = 3.14159

    # 가우시안 필터의 중심 좌표를 계산합니다.
    center = (filter_size - 1) / 2

    for i in range(filter_size):
        for j in range(filter_size):
            # 현재 Iteration (i, j)에서 중심 좌표를 뺀 값을 a와 b에 저장합니다.
            a, b = i - center, j - center

            # 가우시안 함수의 지수 부분을 계산합니다. (자연상수 exp와 다릅니다.)
            exponential = -(a**2 + b**2) / (2 * sigma**2)

            # 2차원 가우시안 함수식을 계산합니다.
            gaussianFilter[i][j] = (1 / (2 * pi * sigma**2)) * (exp**exponential)

    # 생성된 가우시안 필터를 정규화합니다.
    # 모든 요소를 전체 값의 합으로 나누어줍니다.
    filter_sum = sum(sum(gaussianFilter, []))
    gaussianFilter = [[value / filter_sum for value in row] for row in gaussianFilter]

    # 가우시안 필터 (N by N)를 반환합니다.
    return gaussianFilter


# img에서 키포인트를 검출하는 함수입니다.
# N은 주변 이웃 픽셀 경계의 임계값, threshold는 키포인트 검출 기준이 되는 임계값, nonMaxWnd는 non-max suppression window 사이즈입니다.
def FAST(img, N, threshold, nonMaxWnd):
    # Gaussian kernel를 생성합니다.
    gaussianKernel = getGaussianFilter(5, 4)

    # 이미지와 가우시안 커널을 컨볼루션합니다.(가우시안 스무딩)
    img = convolve2d(img, gaussianKernel, mode="same", boundary="fill")

    # 알고리즘에서 사용될 배열을 정의합니다.
    # cross_index는 x축, y축으로의 픽셀 오프셋입니다.
    # circle_index는 중심에서 원형의 픽셀 오프셋입니다. 원형 모양으로 주변 16 픽셀을 나타내고 있습니다.
    crossIndex = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
    circleIndex = np.array(
        [
            [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
            [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1],
        ]
    )

    # 결과적으로 찾아낸 코너를 저장할 배열과 키포인트 좌표를 저장할 빈 리스트를 초기화합니다.
    corner_img = np.zeros(img.shape)
    keypoints = []

    # 이미지 각각의 가장자리에 3픽셀의 여유 공간을 두고 픽셀을 순회합니다.
    for y in range(3, img.shape[0] - 3):
        for x in range(3, img.shape[1] - 3):
            # 현재 픽셀의 밝기값을 pixelIntensity로 저장하고, 임계값 T를 설정합니다.
            pixelIntensity = img[y, x]
            if threshold < 1:
                T = threshold * pixelIntensity
            else:
                T = threshold

            # 현재 픽셀을 중심으로 하는 가로와 세로 방향으로의 밝기 변화를 확인하여 경계로 판단할지 여부를 결정합니다.
            # 조건을 만족하는 픽셀 개수가 최소 3개 이상인지 확인하는 것입니다.
            # 고속 FAST 검사를 적용한 것입니다. 모든 픽셀에 대해 주변 16픽셀을 모두 검사하지 않고 우선 현재 픽셀로부터 (+3,0),(0,+3),(-3,0),(0,-3)을 검사하고 3개 이상이 만족하는지 판단합니다.
            if (
                np.count_nonzero(
                    pixelIntensity + T < img[y + crossIndex[0, :], x + crossIndex[1, :]]
                )
                >= 3
                or np.count_nonzero(
                    pixelIntensity - T > img[y + crossIndex[0, :], x + crossIndex[1, :]]
                )
                >= 3
            ):
                # 조건이 만족될 경우, 16픽셀(circle)를 확인합니다.
                # 조건을 만족하는 픽셀 개수가 N개 이상인지 확인합니다.
                if (
                    np.count_nonzero(
                        img[y + circleIndex[0, :], x + circleIndex[1, :]]
                        >= pixelIntensity + T
                    )
                    >= N
                    or np.count_nonzero(
                        img[y + circleIndex[0, :], x + circleIndex[1, :]]
                        <= pixelIntensity - T
                    )
                    >= N
                ):
                    # 두 조건을 만족하면 이때 키포인트로 결정합니다.
                    # 해당 좌표를 keypoints 리스트에 추가하고, 해당 위치의 corner_img 값을 update합니다.
                    keypoints.append([x, y])
                    corner_img[y, x] = np.sum(
                        np.abs(
                            pixelIntensity
                            - img[y + circleIndex[0, :], x + circleIndex[1, :]]
                        )
                    )

    # Nonmax suppression을 통해 인접한 키포인트와 동일 객체에 대한 복수의 detection을 제거합니다.
    newKeyPoints = []

    # Keypoints 리스트에서 한 좌표씩 꺼냅니다.
    for [x, y] in keypoints:
        # 키포인트 주변 3픽셀을 고려합니다.(저의 테스트케이스에서 항상 window를 3으로 줬기 때문에 이렇게 설명하겠습니다.)
        window = corner_img[
            y - nonMaxWnd : y + nonMaxWnd + 1, x - nonMaxWnd : x + nonMaxWnd + 1
        ]
        # 윈도우 내에서 가장 큰 값의 위치를 찾습니다. (numpy 라이브러리 함수 이용)
        locationMax = np.unravel_index(window.argmax(), window.shape)
        newX = x + locationMax[1] - nonMaxWnd
        newY = y + locationMax[0] - nonMaxWnd
        newKp = [newX, newY]
        # 중복을 체크하고 중복되지 않았다면 새로운 키포인트를 추가합니다.
        if newKp not in newKeyPoints:
            newKeyPoints.append(newKp)

    # NMS를 수행한 keypoint 배열을 반환합니다.
    return np.array(newKeyPoints)


# BRIEF 디스크립터를 계산하는 함수입니다.
# 이미지, 키포인트, 패치 사이즈, 이진비트 수(512를 사용했습니다)를 파라미터로 받습니다.
def BRIEF(img, keypoints, size, n):
    # 시드값을 설정하고 난수를 생성합니다.
    fixedSeed = 42
    random = np.random.RandomState(seed=fixedSeed)

    # Gaussian kernel를 생성합니다.
    gaussianKernel = getGaussianFilter(7, 5)

    # 이미지와 가우시안 커널을 컨볼루션합니다.(가우시안 스무딩)
    img = convolve2d(img, gaussianKernel, mode="same", boundary="fill")

    # -((size - 2) // 2) + 1 ~ (size // 2)까지의 범위에서 (n * 2, 2) 크기의 난수 행렬을 생성합니다.
    samples = random.randint(-(size - 2) // 2 + 1, (size // 2), (n * 2, 2))
    samples = np.array(samples, dtype=np.int32)
    # 생성된 샘플 좌표를 두 부분으로 나누어 각각 pos1과 pos2에 저장합니다.
    pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    # 유효한 키포인트를 선택하기 위한 마스크를 생성합니다.
    # 키포인트가 패치 경계 내에 있는 경우에만 유효한 것으로 간주합니다.
    mask = (
        ((size // 2 - 1) < keypoints[:, 0])
        & (keypoints[:, 0] < (cols - size // 2 + 1))
        & ((size // 2 - 1) < keypoints[:, 1])
        & (keypoints[:, 1] < (rows - size // 2 + 1))
    )

    # 유효한 키포인트의 좌표만 선택하여 keypoints 배열을 업데이트합니다.
    keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)

    #  BRIEF 디스크립터를 저장할 배열을 초기화합니다.
    descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

    # 각 랜덤 샘플 쌍에 대해 BRIEF 디스크립터 계산을 수행합니다.
    # 각 iteration에서는 한 쌍의 샘플 좌표를 사용하여 비교를 수행합니다.
    for pos in range(pos1.shape[0]):
        # 현재 Iteration에서 사용할 첫 번째 샘플 좌표를 posR0와 posC0에 할당합니다.
        posR0 = pos1[pos, 0]
        posC0 = pos1[pos, 1]
        # 현재 Iteration에서 사용할 두 번째 샘플 좌표를 posR0와 posC0에 할당합니다.
        posR1 = pos2[pos, 0]
        posC1 = pos2[pos, 1]
        for key in range(keypoints.shape[0]):
            # 현재 반복에서 사용할 키포인트의 좌표를 keyR과 keyC에 할당합니다.
            keyR = keypoints[key, 1]
            keyC = keypoints[key, 0]
            # 첫 번째 샘플의 픽셀 값과 두 번째 샘플의 픽셀 값을 비교하여 이진 디스크립터를 생성합니다.
            if img[keyR + posR0, keyC + posC0] < img[keyR + posR1, keyC + posC1]:
                descriptors[key, pos] = True

    # descriptor 배열을 반환합니다.
    return descriptors
