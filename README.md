# Comupter Vision 
# TOPIC : Panorama using Homography


---

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/directoryTree.png" width="400" height="auto">
  제가 진행한 프로젝트의 파일 디렉토리 구조입니다. 키포인트 및 디스크립터 검출 등 최대한 bottom부터 라이브러리의 도움을 받지 않고 직접 구현하는 데에 집중했습니다. 그렇기 때문에 코드를 작성하면서 구분되는 기능별로 모듈화했습니다. 파일 내의 함수에 대한 자세한 설명은 소스 코드 내 주석 처리로 적어 놓았습니다. 본 보고서에서는 모듈별 큰 동작 원리와 문제점을 어떤 식으로 해결해 나갔는지 등에 대해서 설명하도록 하겠습니다. 사용한 이미지는 inputs 폴더에 있으며 결과 이미지는 results 폴더에 있습니다.

---

## main.py

  프로그램의 엔트리 포인트입니다. 메인 함수와 getMatch 함수가 작성되어 있습니다. 제가 구현한 이미지 스티칭 프로그램의 추상화된 동작 방식은 이러합니다. 

1. 이미지 파일 두 개를 읽습니다.
2. 각 이미지의 keypoints와 descriptors를 검출합니다.
3. 두 디스크립터를 통해 대응점을 찾습니다.
4. 두 이미지의 관계를 표현한 행렬 homography를 구합니다.
5. 구한 호모그래피 행렬을 기반으로 두 이미지를 stitch합니다. 

  큰 동작 개념은 위와 같지만 중간에 개선점을 지속적으로 추가하였습니다. 뒷 목차에서 설명하도록 하겠습니다.

---

## FAST_and_BRIEF.py

  keypoint와 descriptor 검출을 위해 openCV 라이브러리의 ORB, KAZE 등을 사용하면 약 5 line에 완료할 수 있다는 사실을 알고있지만, CV 과목을 수강하면서 Harris corner 외에 다른 특징점 검출 알고리즘이 무엇이 있는지에 대한 궁금증이 생겼고 라이브러리 도움없이 이미지에서 직접 특징점을 검출해보고 싶었습니다. 따라서 특징점 검출을 위해 FAST, 특징점 주변의 decriptor 계산을 위해 BRIEF 알고리즘을 사용하기로 했습니다. 

### - FAST

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/fast.png" width="400" height="auto">

  FAST(Features from Accelerated Segment Test)는 간단하게 중심 픽셀의 밝기값 ± threshold와 원 모양의 주변 16개의 픽셀을 비교하는 알고리즘입니다. 성능을 더욱 최적화하기 위해 고속 test를 적용했습니다. 모든 픽셀에 대해 16개의 픽셀을 검사하는 것은 처리 속도를 매우 늦출 수 있기 때문에 위 첨부 이미지의 기준으로 p가 중심일 때 1번과 9번, 5번과 13번 픽셀을 검사합니다. p가 코너라면 4개의 픽셀 중 최소 3개는 IntensityPixel + threshold보다 밝거나 IntensityPixel - threshold 보다 어두워야 합니다. 따라서 고속 검사를 통과한 픽셀에 대해서 주변 16픽셀을 검사하는 방식으로 효율적으로 계산할 수 있도록 했습니다.

  FAST-9 와 FAST-12 알고리즘이 주로 사용되는데, 둘의 차이는 특징점으로 판단하는 기준입니다. 이웃 픽셀 중에서 조건을 만족하는 픽셀이 12개 이상일 때 특징점으로 판단할 것인지, 9개 이상일 때 판단할 것인지의 차이를 갖고 있습니다. 처리 속도와 성능 면에서 trade-off를 갖고 있습니다. 저는 12 알고리즘을 사용했습니다.

### - BRIEF

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/brief.png" width="350" height="auto">

  BRIEF(Binary Robust Independent Elementary Features)는 descriptor를 Binary 형태로 표현하는 디스크립터 생성 알고리즘입니다. 검출된 각 키포인트를 기준으로 주변 영역의 픽셀을 다른 픽셀과 비교해서 더 밝은 부분을 찾아 binary 형식(0과 1)으로 저장하는 매커니즘입니다. 예를 들어 descriptor의 결과는 01101000(8bits→2 $^8$=256) 등이 될 수 있습니다.

---

## Homography_RANSAC.py

  픽셀로부터 호모그래피 행렬을 계산하는 getHomography와 RANSAC을 수행하면서 최적의 호모그래피 행렬을 찾는데 필요한 useHomography 함수가 정의되어 있습니다. 많은 특징점 중 Outlier를 제거하고, 최적 호모그래피 행렬을 얻는 RANSAC 함수도 정의했습니다. RANSAC에 대해서는 뒷 부분에서 추가 설명하겠습니다.

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/homographyPoints.png" width="auto" height="200">
<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/homographySVD.png" width="auto" height="200">

  Image stitching에서 호모그래피 행렬은 두 이미지 간의 관계를 나타내는 요소를 가진 3x3 행렬이며 한 이미지의 좌표를 다른 이미지의 좌표로 mapping하는데 사용됩니다. 즉, 이 말은 이미지 스티칭의 결과가 좋으려면 첫 째로 코너를 잘 검출해야 하며 결론적으로는 좋은 호모그래피 행렬을 찾아야 한다는 것을 의미합니다.

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/homography.png" width="400" height="auto">

  Projective transformation에서 호모그래피는 이렇게 표현할 수 있습니다. 좌표 성분 A에 대해 A$^T$A의 가장 작은 고유값을 가지는 고유 벡터가 해가 됩니다. 

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/homographyMatrix.png" width="400" height="auto">

⇒ 직접 구현한 호모그래피 행렬을 출력한 모습입니다.

---

## Stitch.py

  이미지 스티칭을 구현하는데 필요한 함수가 정의되어 있습니다. 이전 과정에서 구한 최적의 원근 변환 행렬 H와 이미지 두 개를 사용하여 두 이미지를 붙이는 과정을 수행합니다. 해당 함수를 사용해서 결과 이미지의 사이즈를 측정할 수 있습니다. 또한 톤 매핑을 통해 자연스러운 결과 이미지를 생성하는 모듈입니다. 각 함수에 대한 자세한 설명은 소스 코드 내부에 있습니다.

---

## Visualization.py

  시각화와 관련된 모듈입니다. 프로젝트를 진행하면서 중간 중간에 눈으로 결과를 확인하기 위해 별도로 정의한 함수가 있습니다.

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/visualizationHairshop.png" width="400" height="auto">

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/visualizationCafe.png" width="400" height="auto">


  openCV 라이브러리의 drawMatches 함수를 사용하지 않고, 두 이미지 각각의 feature point 중 서로 어떤 것이 매핑(대응)되었는지 두 이미지를 직접 붙여서 시각화하는 visualize_matches 함수를 직접 구현했습니다. 대응점을 잇는 라인을 잘 구별하기 위해 난수 컬러값을 사용했습니다. 이미지1은 특징점이 파란색으로, 이미지2는 빨간색으로 표시했습니다.

| image1 (Left) | image2 (Right) |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/img1KP.png" width="300" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/img2KP.png" width="300" height="350"> |


  각 이미지에서 검출된 keypoint를 이미지에 표시하는 drawKeypoints 함수를 정의했습니다. 초록색 원 형태로 이미지 픽셀에 표시했습니다. (두 이미지는 다른 이미지입니다. image1에 대한 키포인트 마킹과 image2에 대한 키포인트 마킹 결과입니다.)

| image1 (Left) | image2 (Right) |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/threshold03.png" width="300" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/threshold06.png" width="300" height="350"> |

  FAST 함수 호출 파라미터로 threshold를 조정하여 더 적거나 많은 Feature point를 검출할 수 있습니다. 동일한 이미지에 대해 왼쪽은 threshold를 0.3, 오른쪽은 threshold를 0.6으로 했을 때의 결과입니다. 임계값을 낮게 줬을 때 코너로 검출한 특징점이 더 많다는 것을 확인할 수 있습니다.

---

# 개선 방안

  제가 프로젝트를 지속적으로 진행해오면서 발견한 문제점을 개선한 방안에 대해 설명하도록 하겠습니다.

## 1. Non-Max Suppression

### 1-1. 문제점 파악

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeNMSCafe.png" width="400" height="auto">

- NonMax Suppression X

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeNMS.png">

  구현한 FAST 특징점 검출 알고리즘이 너무 많은 특징점을 검출하여 실행 속도가 느려지는 문제를 겪었습니다. 결과 출력에서 1532개의 특징점이 검출됐다고 했으나 사진에 mark된 좌표가 생각보다 많지는 않았습니다. 이로써 중복 검출한 복수의 동일한 키포인트가 있거나 매우 인접한 키포인트가 검출이 됐을 것이라고 판단했습니다.

### 1-2. 해결 방법

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/afterNMSCafe.png" width="400" height="auto">

- NonMax Suppression O

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/afterNMS.png">

  단지 Threshold를 조절하는 것으로는 중복 키포인트 검출에 한계가 있기 때문에 해당 문제를 해결하기 위해서 CV 수강 중에 배운 것을 활용해보고자 NonMax Suppression을 Default로 수행하도록 했습니다. 적용한 결과 이미지에서 확인할 수 있듯이 동일 이미지에 대해 NMS를 수행했을 때, 훨씬 더 적지만 중요한 키포인트를 제대로 검출하는 모습입니다. image1에 대한 특징점이 1532개 검출됐었던 1-1에 비해서 457개로 줄은 모습입니다.

---

## 2. Gaussian Smoothing

### 2-1. 문제점 파악

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeGaussianCafe.png" width="400" height="auto">

- Gaussian smoothing X (with NMS)

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeGaussian.png">

  가우시안 스무딩을 하지 않은 이미지에서 FAST, BRIEF를 수행한 결과입니다. 처리할 이미지에서 스무딩을 하지 않으면 많은 노이즈가 포함되어 있기 때문에 잘못된 키포인트 검출이 매우 많이 발생합니다.

### 2-2. 해결 방법

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/gaussian.png" width="400" height="auto">

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/afterSmoothing.png">

- Gaussian smoothing O (with NMS)

  FAST, BRIEF 함수 내부에서 이미지를 처리하기 전에 이미지에 가우시안 필터를 컨볼루션하여 스무딩 전처리를 수행했습니다. 목적은 노이즈 제거이며 불필요한 Outlier 검출을 줄일 수 있습니다. 동작을 위한 getGaussianFilter 함수를 정의했습니다.

  동일한 Threshold 기준, 2-1의 전처리를 하지 않은 image1에서 검출된 특징점이 5897개인 것에 비해, 스무딩을 적용한 image1에서 검출된 특징점은 457개로 줄었습니다.

---

## 3. RANSAC

### 3-1. 문제점 파악

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeRANSACMega.png" width="400" height="auto">

- RANSAC X

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeRANSAC.png">

⇒ 이미지1에서 429개, 이미지2에서 284개의 특징점이 검출됐습니다. 각 특징점 사이에 17개가 대응된 것입니다.

  이미지 스티칭에서 결국 가장 중요한 것은 최적의 원근 변환 행렬을 찾는 것이었습니다. 결과 이미지와 직결되기 때문입니다. 좋은 호모그래피 행렬을 찾기 위해서는 잘못된 두 대응점(Outlier)을 최대한 제거해야 하기 때문에 RANSAC을 적용하기로 했습니다. 위 결과 이미지는 RANSAC을 수행하지 않고 대응점을 나타낸 모습입니다. 잘못 대응된 Outlier가 많은 모습을 확인할 수 있습니다.

### 3-2. 해결 방법

  기본적인 동작 방식은 다음과 같습니다.

1. 각 이미지에서 검출된 keypoint 사이의 대응점을 찾습니다. (FAST, BRIEF, getMatch)
    - 잘못된 대응점(outlier)을 포함하고 있다는 가정으로 설명하겠습니다.
2. 파라미터로 받은 반복 횟수 N에 대해 RANSAC을 수행합니다.
    1. 매칭된 특징점 중 4개를 랜덤으로 선택합니다.
        - 4개 미만이라면 프로그램이 종료됩니다. 이때 FAST Threshold를 낮춰서 특징점을 많이 검출해야 합니다.
    2. 4개에 대해 Source 이미지로부터 Destination 이미지로의 원근 변환 행렬을 구합니다. (getHomography)
    3. 구한 H 행렬과 Source 이미지의 특징점 좌표를 이용하여 예측된 Destination 좌표를 구합니다. (useHomography)
    4. 실제 Destination의 특징점과 예측된 좌표를 비교합니다.
        - 비교 기준으로 SSD를 사용합니다.
    5. SSD값이 파라미터로 받은 임계값보다 작으면 inlier로 간주합니다.
3. 추려진 Source inlier과 Destination inlier 사이의 원근 변환 행렬을 다시 구합니다. (새로운 inlier 파라미터로 getHomography 재호출)

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/afterRANSACMega.png" width="400" height="auto">

- RANSAC O 

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/afterRANSAC.png">

  3-1에서 매칭된 17개의 대응점에 대해 3000번 반복 RANSAC을 수행한 결과로 8개의 대응점을 얻었습니다. 매칭 이미지에서도 Outlier가 사라진 모습입니다. Outlier를 배제한 상태에서 H를 다시 구했기 때문에 더 정확한 원근 변환 행렬을 얻습니다.

---

## 4. Tone mapping

### 4-1. 문제점 파악

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeImproveToneMapping.png" width="400" height="auto">

  카메라의 각도나 찍는 시점이 살짝만 바뀌어도 두 이미지의 색감, 명암 등이 다르기 때문에 스티칭된 결과 이미지가 매끄럽지 않았습니다. 

### 4-2. 해결 방법

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/ImproveToneMapping.png" width="400" height="auto">

  톤 매핑 알고리즘을 적용하여 전체적으로 매끄러운 이미지가 출력되도록 했습니다. 완벽하게 두 이미지의 색감을 일치시킬 수는 없지만 이전 4-1 스티칭 이미지에 비해 자연스러워진 모습입니다.

---

## 5. Speed Up Processing

### 5-1. 문제점 파악

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/beforeImproveTime.png">

- Resizing X (original image 사용)

  제가 직접 휴대폰 카메라로 샘플 이미지를 촬영했기 때문에 해상도가 높은 고용량 이미지가 사용됐습니다. 원본 이미지에 대해 모든 계산과 처리를 하기 때문에 실행속도가 매우 느렸습니다. 

### 5-2. 해결 방법

<img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/improveTime.png">

- Resizing O (resized image 사용)

  따라서 저는 프로젝트를 진행하는데 있어서 빠른 피드백과 수정이 필요했기 때문에 결과적으로 스티칭된 이미지의 해상도를 조금 희생하고 처리 속도를 줄이기 위해서 원본 이미지를 리사이징하여 계산에 사용했습니다. 5-1보다 4배 가까이 빨라진 모습입니다.

---

# 진행 결과

| KeyPoints | Result |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/megaMatched.png" width="500" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/megaResult.jpeg" width="400" height="350"> |

- resize scale : 0.5
- FAST threashold : 0.3
- RANSAC : 3000, RANSAC_Threshold : 30

<br/>

| KeyPoints | Result |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/starbucksMatched.png" width="500" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/starbucksResult.jpeg" width="400" height="350"> |

- resize scale : 0.5
- FAST threashold : 0.4
- RANSAC : 3000, RANSAC_Threshold : 30

<br/>

| KeyPoints | Result |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/cafeMatched.png" width="500" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/cafeResult.jpeg" width="400" height="350"> |

- resize scale : 0.5
- FAST threashold : 0.3
- RANSAC : 3000, RANSAC_Threshold : 30

<br/>

| KeyPoints | Result |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/hairshopMatched.png" width="500" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/hairshopResult.jpeg" width="400" height="350"> |

- resize scale : 0.5
- FAST threashold : 0.4
- RANSAC : 3000, RANSAC_Threshold : 30

<br/>

| KeyPoints | Result |
|:-----------:|:-------------:|
| <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/wayMatched.png" width="500" height="350"> | <img src="https://github.com/ssongms/Panorama-with-Homography/blob/main/assets/wayResult.jpeg" width="400" height="350"> |

- resize scale : 0.5
- FAST threashold : 0.4
- RANSAC : 3000, RANSAC_Threshold : 30

