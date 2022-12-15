# Autonomous Reading of Gauges in Unstructured Environments
# SENSORS - Edoardo Milana, ect 2 peoples
import cv2
import math


############################################################################################################
## OpenCV 파트
def gauge_cv_adaptiveThreshold(gray_img, threshold=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY_INV, C=7.0):
  """
  >>> cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)

  Parameters:	
    src => grayscale image
    maxValue => 임계값
    adaptiveMethod => thresholding value를 결정하는 계산 방법
    thresholdType => threshold type
    blockSize => thresholding을 적용할 영역 사이즈
    C => 평균이나 가중평균에서 차감할 값

    https://meet.google.com/pui-hnty-adt
  """
  h, w = gray_img.shape
  s = int(w / 8)
  gray = cv2.adaptiveThreshold(gray_img, 
                            threshold, 
                            adaptiveMethod,
                            thresholdType, s, C)
  return gray
def gauge_cv_morphology(gray_img, 
                      morphology_operator=cv2.MORPH_CLOSE, 
                      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))):
  return cv2.morphologyEx(gray_img, morphology_operator, kernel)
def gauge_cv_findContours(gray_img, retreival_mode=cv2.RETR_CCOMP, approximation_mode=cv2.CHAIN_APPROX_NONE):
  cnts, hier = cv2.findContours(gray_img, retreival_mode, approximation_mode)
  return cnts, hier
def gauge_cv_drawContours(overlap_img, cnts):
  cv2.drawContours(overlap_img, cnts, -1, (0, 0, 255), 2)
  return True
def gauge_cv_findBestEllipseCandidateIndex(img, cnts):
  """
  parameters:
   - img: 이미지
   - cnts: 컨투어 후보
  """
  length = 1


############################################################################################################
## 의사결정 파트
def gauge_decision_getTopTenContourIndexes(data) -> list[int]:
  """
  parameter:
   - data :findContours로 출력된 컨투어 정보 데이터를 입력으로 한다.

  컨투어 포인트가 많은 상위 10% 데이터를 반환한다.
  """
  hash_map = {}
  length = len(data)
  top_ten = int(length * 0.1)

  # 해시맵
  for i, d in enumerate(data):
    v = len(d)
    if hash_map.get(v) is not None:
      hash_map[len(d)].append(i)
    else:
      hash_map[len(d)] = [i]
  
  # 키 정렬
  keys = sorted(hash_map.keys(), reverse=True)
  
  # index 조회 완료
  count = 0
  index_list = []
  for key in keys:
    for k in hash_map[key]:
      if count >= top_ten:
        return index_list
      index_list.append(k)
      count += 1
  return []
def gauge_decision_isValidEliipse(h, w, ellipse, skew_threshold=0.4) -> bool:
  """
  return:
   - false: 부적합한 타원
   - true: 적합한 타원

  parameter:
    h: 이미지 높이
    w: 이미지 넓이
    ellipse = cv2.fitEllipse(cnts)
     - ellipse[0] : (center x, center y)
     - ellipse[1] : (width, height)
     - ellipse[2] : rotation angle
    - len(ellipse) == 3

  1. skew factor 체크
  >>> | ellipse_width - ellipse_height| / (ellipse_width + ellipse_height)
  >>> 0에 가까울 수록 원에 가까워짐
  >>> 0.4 보다 작은 경우만 허용

  2. area factor 체크
  >>> (ellipse_width * ellipse_hiehgt) / (image_width * image_height)
  >>> 1에 가까울수록 타원이 이미지 크기에 가까워진다.
  >>> 0.2 ~ 0.8 사이만 허용한다.

  3. central factor 체크
  >>> 
  """
  d = (h + w) / 2
  if (abs(ellipse[1][0] - ellipse[1][0]) / (ellipse[1][0] + ellipse[1][0])) > skew_threshold:
    return False
  if ((ellipse[1][0] * ellipse[1][1]) / (h * w)) > 0.8 or ((ellipse[1][0] * ellipse[1][1]) / (h * w)) < 0.2:
    return False
  if math.sqrt((ellipse[0][0] - w/2) ** 2 + (ellipse[0][1] - h/2) ** 2) > (d * 0.2):
    return False
  return True
def gauge_decision_getLongestContourIndex(cnts, valid_index: list[int]) -> int:
  length_cache = 0
  index_cache = 0
  for index in valid_index:
    if length_cache < len(cnts[index]):
      length_cache = len(cnts[index])
      index_cache = index
  return index_cache












