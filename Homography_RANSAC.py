"""

* Homography 계산, RANSAC 알고리즘 구현과 관련된 모듈입니다.

"""


import numpy as np
import random


# 파라미터로 받은 두 점을 기반으로 호모그래피 행렬을 계산하고 반환하는 함수입니다.
def getHomography(srcPoint, dstPoint):
    # 호모그래피 행렬의 요소를 계산하기 위해 빈 리스트를 생성합니다.
    H = []

    # 대응점 수를 N에 저장합니다.
    N = srcPoint.shape[0]

    # 계산 용이성을 위해 numpy 배열로 바꿉니다.
    srcArr = np.asarray(srcPoint)
    dstArr = np.asarray(dstPoint)

    for n in range(N):
        # 각각의 대응점의 src 좌표로 호모그래피 행렬의 각 요소를 계산하여 H 리스트에 추가합니다.
        #
        src = srcArr[n]
        H.append(-src[0])
        H.append(-src[1])
        H.append(-1)
        H.append(0)
        H.append(0)
        H.append(0)

    # H를 numpy 배열로 변환하고
    # H의 형상을 (2 * N, 3)으로 변경하여 H_1에 저장합니다.
    H = np.asarray(H)
    H_1 = H.reshape(2 * N, 3)

    # 0으로 초기화된 2차원 배열 H_2를 생성합니다.
    H_2 = np.zeros([2 * N, 3], dtype=int)

    # H_2의 특정 부분에 H_1을 reverse해서 복사합니다.
    for i in range(0, 2 * N, 2):
        H_2[i : i + 2, 0 : i + 3] = np.flip(H_1[i : i + 2, 0 : i + 3], axis=0)

    # H_1, H_2를 수평으로 연결하여 H_3에 저장합니다.
    H_2 = np.asarray(H_2)
    H_3 = np.concatenate((H_1, H_2), axis=1)

    # 호모그래피 행렬의 추가 요소를 계산하기 위해 빈 리스트를 생성합니다.
    H_4 = []
    for n in range(N):
        # 대응점 src와 dst를 기반으로
        # 호모그래피 행렬의 추가 요소를 계산하여 H_4 리스트에 추가합니다.
        source = srcArr[n]
        destination = dstArr[n]
        H_4.append(source[0] * destination[0])
        H_4.append(source[1] * destination[0])
        H_4.append(destination[0])
        H_4.append(source[0] * destination[1])
        H_4.append(source[1] * destination[1])
        H_4.append(destination[1])

    # H_4의 형상을 (2 * N, 3)으로 변경합니다.
    H_4 = np.asarray(H_4)
    H_4 = H_4.reshape(2 * N, 3)

    # H_3, H_4를 수평으로 연결하여 H_5에 저장합니다.
    H_5 = np.concatenate((H_3, H_4), axis=1)

    # H_5의 전치 행렬과 자신(H_5)의 행렬 곱을 계산하여 H8을 생성합니다.
    H_8 = np.matmul(np.transpose(H_5), H_5)

    # 행렬 H_8의 고유값과 고유벡터를 계산하여 w, v에 각각 저장합니다.
    w, v = np.linalg.eig(H_8)

    # 고유값 중 최소값을 minumum에 저장합니다.
    minimum = w.min()

    # 최소 고유값의 고유벡터를 찾고 H_matrix에 저장합니다.
    for i in range(len(w)):
        if w[i] == minimum:
            H_matrix = v[:, i]

    # H_matrix의 형상을 (3,3)으로 변경합니다.
    H_matrix = np.asarray(H_matrix)
    H_matrix = H_matrix.reshape(3, 3)

    # 정규화 : 호모그래피 행렬을 마지막 요소로 나눕니다.
    H_matrix = H_matrix / H_matrix[2, 2]

    # 계산이 완료된 호모그래피 행렬을 반환합니다.
    return H_matrix


# 좌표 배열에 호모그래피 변환 매트릭스를 적용하여 변환된 좌표를 반환하는 함수입니다.
def useHomography(src, Homography):
    # 호모그래피를 적용한 결과로 얻어지는 점들을 저장할 빈 리스트 초기화합니다
    # 그리고 입력 점들의 개수를 N에 저장합니다.
    output = []
    Num = src.shape[0]

    # src의 각 row에 대해 루프를 돕니다.
    for row in src:
        # 현재 행의 x 및 y 좌표를 [x, y, 1] 형태의 행렬로 만듭니다.
        # (호모그래피 변환을 위한 확장된 좌표)
        input = np.matrix([row[0, 0], row[0, 1], 1])

        # 행렬을 전치하여 열 벡터로 변환합니다.
        input = input.transpose()

        # 호모그래피 매트릭스를 이용하여 변환을 수행합니다.
        mappedPoints = np.matmul(Homography, input)

        # 변환된 점의 확장된 좌표를 정규화하는 과정입니다.
        # 그리고 이 좌표를 결과 리스트에 추가합니다.
        output.append(mappedPoints[0] / mappedPoints[2])
        output.append(mappedPoints[1] / mappedPoints[2])

    # 반환 형식에 맞게 형태를 변경합니다.
    output = np.asarray(output)
    output = output.reshape(Num, 2)

    # 결과를 반환합니다.
    return output


def RANSAC(srcX, dstX, repeatCount, threshold):
    """
    * 데이터 중 Outlier를 제거하기 위한 RANSAC 함수입니다.
    * Source image와  Destination image에서 매치된 좌표와 반복 횟수, 그리고 임계값을 파라미터로 받습니다.
    * 최적의 inlier index와 최적 호모그래피 행렬을 반환합니다.
    """

    # 대응점의 개수를 n에 저장합니다.
    # srcX와 dstX의 크기는 같습니다. 따라서 둘 중 아무거나 선택해서 초기화해줍니다.
    n = srcX.shape[0]

    # RANSAC을 수행하기 위한 최소 조건입니다. (예외 처리)
    if n < 4:
        print("Error : RANSAC을 수행하기 위해 최소 4개의 대응점이 필요합니다.")
        exit()

    # 호모그래피 행렬 H를 0으로 초기화합니다. (3 by 3)
    H = np.zeros([3, 3])

    # 각 반복(테스트)에서 inlier의 인덱스와 개수를 저장할 리스트를 생성합니다.
    inlierIndex = []
    inlierCount = []

    # loop counter를 설정합니다.
    current = 0

    # 파라미터로 받은 반복횟수만큼 RANSAC이 동작합니다.
    while current < repeatCount:
        # iteration마다 초기화되는 inlier_id 배열입니다.
        # 즉 inlierIndex와 inlierCount에 append하기 위한 임시 배열입니다.
        inliers_id = []

        # 대응점 중 랜덤으로 4개를 선택합니다.
        pts_index = random.sample(range(0, n), 4)

        # Xs와 Xd에서 선택된 포인트를 저장할 리스트입니다.
        newSrc = []
        newDst = []

        # 랜덤으로 선택된 포인트를 각각 저장합니다.
        for i in range(4):
            newSrc.append(srcX[pts_index[i]][:])
            newDst.append(dstX[pts_index[i]][:])

        # 선택된 대응점을 각각 행렬로 변환합니다.
        newSrc = np.asarray(newSrc)
        newDst = np.asarray(newDst)
        newSrc = np.asmatrix(newSrc)
        newDst = np.asmatrix(newDst)

        # 선택된 포인트들을 기반으로 호모그래피 행렬을 계산합니다.
        H = getHomography(newSrc, newDst)

        # 추정된 호모그래피 행렬을 사용해서 srcX를 변환한 예측 좌표를 계산합니다.
        # 예측 좌표와 실제 좌표의 차이(SSD)를 확인하는 로직의 알고리즘입니다.
        srcX = np.asmatrix(srcX)
        predictedDst = useHomography(srcX, H)

        # SSD (Sum of Squared Differences)를 기반으로 inlier를 확인합니다.
        for i in range(n):
            SSD = (round(predictedDst[i][0]) - int(dstX[i, 0])) ** 2 + (
                round(predictedDst[i][1]) - int(dstX[i, 1])
            ) ** 2
            if SSD < threshold:  # 임계값보다 작으면 해당 인덱스를 inliers_id에 추가합니다.
                if i not in inliers_id:
                    inliers_id.append(i)

        # 현재 iteration에서의 inlier 인덱스와 개수를 저장합니다.
        inlierIndex.append(inliers_id)
        inlierCount.append(len(inliers_id))

        # loop count를 중가시킵니다.
        current = current + 1

    # inlier의 개수가 가장 많았던 루프에서의 인덱스를 찾습니다.
    # 찾은 후 inlier의 index들을 저장합니다.
    maxCountIdx = inlierCount.index(max(inlierCount))
    optimized_inlier_idx = inlierIndex[maxCountIdx]

    # 최적의 inlier index에 해당하는 대응점을 각각 저장합니다.
    # 저장하기 위한 배열을 새로 생성합니다.
    src_inlier = []
    dst_inlier = []
    for i in optimized_inlier_idx:
        src_inlier.append(srcX[i][:])
        dst_inlier.append(dstX[i][:])

    # 대응점들을 각각 행렬로 변환합니다.
    src_inlier = np.asarray(src_inlier)
    dst_inlier = np.asarray(dst_inlier)
    src_inlier = np.asmatrix(src_inlier)
    dst_inlier = np.asmatrix(dst_inlier)

    # 최종적으로 선택된 inlier 대응점을 사용해서 최적의 호모그래피 행렬 H를 계산합니다.
    H = getHomography(src_inlier, dst_inlier)

    # 최종 계산된 최적 inlier 인덱스와 호모그래피 행렬을 반환합니다.
    return optimized_inlier_idx, H
