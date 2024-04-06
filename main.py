"""

* 프로그램의 엔트리 포인트입니다.

"""

import numpy as np
from scipy.spatial.distance import cdist
import cv2
import time
import os

# 직접 구현한 모듈 import
from FAST_and_BRIEF import FAST, BRIEF
from Visualization import visualize_matches, drawKeypoints
from Homography_RANSAC import RANSAC
from Stitch import stitch


def getMatch(desc1, desc2):
    # scipy의 cdist 함수를 사용하여 두 디스크립터 간의 해밍 거리를 계산합니다.
    distance = cdist(desc1, desc2, metric="hamming")

    # idx1은 desc1의 인덱스 배열을 저장합니다.
    # idx2는 distance의 각 행에서 최솟값을 가지는 열의 index를 저장합니다. (np.argmin 사용)
    idx1 = np.arange(desc1.shape[0])
    idx2 = np.argmin(distance, axis=1)

    # cross check를 수행해서 두 이미지 간에 대응하는 지점을 찾습니다.
    # match는 distance의 각 열에서 최솟값을 가지는 행의 index를 저장합니다.
    # => match는 desc2의 각 특징점에 대해 거리가 가장 가까운 desc1의 특징점의 인덱스를 저장합니다.
    # idx1과 idx2는 해밍 거리를 기반으로 매칭된 특징점들의 인덱스를 갖습니다.
    match = np.argmin(distance, axis=0)
    mask = idx1 == match[idx2]
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # first_feature는 각 행에 대해 최소 거리를 계산하고, second_feature는 첫 번째 최소 거리를 제외한 나머지 최소 거리를 계산합니다.
    newDistance = distance
    first_feature = np.min(newDistance[idx1, :], axis=1)
    newDistance[idx1, idx2] = np.inf
    second_feature = np.min(newDistance[idx1, :], axis=1)
    # 거리 비율이 0.5보다 작은 경우, 해당 인덱스를 남기고 나머지는 저장하지 않습니다.
    mask = first_feature / second_feature <= 0.5
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # 거리에 따라 정렬하고 sortedIdx에 저장합니다.(argsort)
    # 정렬된 인덱스를 기반으로 최종 대응 배열을 생성합니다.
    dist = distance[idx1, idx2]
    sortedIdx = dist.argsort()
    result_matches = np.column_stack((idx1[sortedIdx], idx2[sortedIdx]))

    # 최종 결과를 반환합니다.
    return result_matches


if __name__ == "__main__":
    start_time = time.time()
    # 이미지를 read합니다.(BGR)
    originImg1 = cv2.imread("inputs/way1.jpg")
    originImg2 = cv2.imread("inputs/way2.jpg")

    # 처리 용이성을 위해 원본 이미지를 Grayscale로 바꿔서 grayimg1, grayimg2에 각각 저장합니다.
    grayimg1 = cv2.cvtColor(originImg1, cv2.COLOR_BGR2GRAY)
    grayimg2 = cv2.cvtColor(originImg2, cv2.COLOR_BGR2GRAY)

    # resize 크기를 지정해줍니다.
    resize_scale = 0.5

    # 처리에서 사용할 grayimg1, grayimg2를 리사이징합니다.
    # 처리 속도가 비약적으로 상승합니다.
    img1 = cv2.resize(grayimg1, None, fx=resize_scale, fy=resize_scale)
    img2 = cv2.resize(grayimg2, None, fx=resize_scale, fy=resize_scale)

    # 구하는 좌표, 행렬 등이 다 리사이징된 이미지를 기준으로 처리하기 때문에 원본 이미지또한 리사이징을 진행합니다.
    originImg1 = cv2.resize(originImg1, None, fx=resize_scale, fy=resize_scale)
    originImg2 = cv2.resize(originImg2, None, fx=resize_scale, fy=resize_scale)

    # 각 이미지에 대한 키포인트, 디스크립터를 계산합니다.
    # FAST(image, N, threshold, nonmax suppression window)를 호출합니다.
    # FAST_and_BRIEF.py에 정의되어 있습니다.
    kp1 = FAST(img1, 12, 0.4, 3)
    kp2 = FAST(img2, 12, 0.4, 3)

    # patch size = 9, n = 256
    firstImg_descriptor = BRIEF(img1, kp1, 9, 256)
    secondImg_descriptor = BRIEF(img2, kp2, 9, 256)

    # 각각의 descriptor를 통해 매칭을 계산합니다.
    result_matches = getMatch(firstImg_descriptor, secondImg_descriptor)

    # image1과 image2에서 각각 검출된 keypoint의 개수를 출력합니다.
    print(f"Number of keypoints of first_image: {len(kp1)}")
    print(f"Number of keypoints of second_image: {len(kp2)}")

    # 두 이미지 사이에 매칭된 좌표의 개수를 출력합니다.
    print(f"Number of result_matches: {len(result_matches)}")

    # 키포인트를 mark한 이미지를 시각화해서 보여줍니다. (drawKeypoints는 Visualization.py에 정의되어 있습니다.)
    result_img1 = drawKeypoints(originImg1, kp1)
    result_img2 = drawKeypoints(originImg2, kp2)
    cv2.imshow("Image1 Marked", result_img1)
    cv2.imshow("Image2 Marked", result_img2)

    # 각 이미지에서의 match 픽셀을 서로 다른 배열에 저장합니다.
    matched_points1 = np.array([kp1[idx] for idx, _ in result_matches])
    matched_points2 = np.array([kp2[idx] for _, idx in result_matches])

    # 잘못된 Match(Outlier)를 제거하기 위해 RANSAC을 수행합니다.
    # 최적화된 inlier 정보와 최적화된 호모그래피 매트릭스를 얻을 수 있습니다.
    # 뒤의 두 인자는 반복 횟수와 threshold 설정값 입니다. (Homography_RANSAC.py에 정의되어 있습니다.)
    inlier_index, H = RANSAC(matched_points1, matched_points2, 3000, 30)
    print("Number of result_matches after RANSAC : ", len(inlier_index))

    # 최적화된 매칭을 담기위한 배열을 생성하고 inlier 좌표를 넣어줍니다.
    newMatches = []
    for i in range(len(inlier_index)):
        newMatches.append(result_matches[inlier_index[i]])

    # 최종 매칭을 시각화해서 보여줍니다. (아직 stitch를 하기 전입니다.)
    # visualization.py에 정의되어 있습니다.
    visualize_matches(originImg1, kp1, originImg2, kp2, newMatches)

    # 두 이미지를 stitch합니다. (Stitch.py에 정의되어 있습니다.)
    finalImg = stitch(originImg2, originImg1, H)

    # 프로그램 동작 시간을 체크합니다.
    end_time = time.time()
    print(f"소요 시간: {end_time - start_time} 초")

    # 최종 결과 이미지를 시각화합니다.
    cv2.imshow("Stitched(Panorama) with tone mapping", finalImg)

    # 결과 이미지를 outputs 폴더에 저장하기 위해 정규화합니다.
    # 저장하기 전에 이미지의 픽셀 값 범위를 0에서 255로 조절해야 합니다.
    output_path = "./outputs"
    output_file_path = os.path.join(output_path, "way_result.jpg")
    finalImg = cv2.normalize(finalImg, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_file_path, finalImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
