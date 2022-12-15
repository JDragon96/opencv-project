# https://datascience.stackexchange.com/questions/69397/ellipses-detecting-at-the-image
# https://076923.github.io/posts/Python-opencv-22/
# https://rohankjoshi.medium.com/the-equation-for-a-rotated-ellipse-5888731da76
# https://stackoverflow.com/questions/62698756/opencv-calculating-orientation-angle-of-major-and-minor-axis-of-ellipse

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
# from collections import OrderedDict

def largestCircleIndex(circles):
  largestIndex = 0
  preR = 0
  for index, circle in enumerate(circles):
    if preR < circle[2]:
      largestIndex = index
      preR = circle[2]
  return largestIndex

def LongestContourIndex(data):
  index = 0
  l = 0
  for i, d in enumerate(data):
    if l < len(d):
      print(len(d))
      l = len(d)
      index = i
  return index

def GetBestTenData(data):
  hash_map = {}

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
      if count >= 10:
        return index_list
      index_list.append(k)
      count += 1
  return []
      
def ShowPoints(data, title):
  points = data.reshape(-1, 2)
  plt.plot(points[:, 0], -points[:, 1], '.', color='k')
  plt.title(title)
  plt.show()
      
def methods1():
  img = cv2.imread("./data/clean_gauge1.jpg", cv2.IMREAD_COLOR)
  cv2.imshow("hi", img)
  cv2.waitKey()
  
  dst = img.copy()
  gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 10, minRadius = 80, maxRadius = 120)
  cv2.imshow("hi", gray)
  cv2.waitKey()

  print(circles)

  # for i in circles[0]:
  #   cv2.circle(dst, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 0), 5)
  i = circles[0][largestCircleIndex(circles[0])]
    
  cv2.circle(dst, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 0), 5)
  cv2.imshow("dst", dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def EllipseEq3(long, short, angle, center_x, center_y, x, y):
  a = np.deg2rad(angle)
  x = x - center_x
  y = y - center_y
  p1 = ((long**2) * (math.sin(a) ** 2) + (short**2) * (math.cos(a) ** 2) * (x ** 2)) + 2 * (short ** 2 - long ** 2)* math.sin(a) * math.cos(a) * x * y + ((long ** 2) * (math.cos(a) ** 2) + (short**2) * (math.sin(a) ** 2)) * (y**2)
  p2 = (long**2) * (short**2)
  return p1 / p2
  


def EllipseTest():
  img = cv2.imread("./data/clean_gauge1.jpg", cv2.IMREAD_COLOR)
  img_copy = img.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  h, w = gray.shape
  s = int(w / 8)
  gray = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, s, 7.0)
  
  # Morphology opening
  gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
  cv2.imshow("윤곽선 뭉뚱그리기", gray)
  cv2.waitKey()
  
  # 윤곽선 찾기
  cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
  imgs = cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
  
  # TODO: findContour()에서 hier의 역할은?
  # OK
  # for i, cont in enumerate(cnts):
  #   h = hier[0, i, :]
  #   if len(cnts[i]) < 5: continue
    
  #   # TODO: fitEllipse()에서 리턴 결과 해석은 어떻게?
  #   # 
  #   if h[3] != -1:
  #     elps = cv2.fitEllipse(cnts[i])
  #   elif h[2] == -1:
  #     elps = cv2.fitEllipse(cnts[i])
  #   print(elps)
  #   cv2.ellipse(imgs, elps, (0, 255, 0), 2)
  # cv2.imshow("Output image", imgs)
  # cv2.waitKey()
  
  # for cont in cnts:
    
  
  # index = LongestContourIndex(cnts)

  
  # imgs = cv2.circle(imgs, (int(elps[0][0]), int(elps[0][1])), 5, (255,0,0), -1)
  # imgs = cv2.circle(imgs, (int(elps[1][0]), int(elps[1][1])), 5, (255,255,0), -1)
  indexes = GetBestTenData(cnts)
  print(img.shape)
  h = imgs.shape[0]
  w = imgs.shape[1]
  target = []
  
  d = (h + w) / 2
  
  # 1차 후보 판단.
  for i in indexes:
    elps = cv2.fitEllipse(cnts[i])  
    # Skew
    if (abs(elps[1][0] - elps[1][1]) / (elps[1][0] + elps[1][1])) > 0.4:
      continue
    
    # 특정 크기 이상만 사용한다.
    if ((elps[1][0] * elps[1][1]) / (h * w)) > 0.8 or ((elps[1][0] * elps[1][1]) / (h * w)) < 0.2:
      continue
    print("통과2")
    
    # 게이지가 화면 중앙에 위치하도록 함
    if np.sqrt((elps[0][0] - w/2) ** 2 + (elps[0][1] - h/2) ** 2) > (d * 0.2):
      continue
    print("통과3")
    
    # TODO: 뽑힌 포인트 기반으로, 원의 형상을 띄는 최적의 후보 판단 필요함      
    # SVD 기반으로 판단해보기
    # 오히려 게이지 껍질을 전부 포함한 게 원에 가까웠음
    U, S, VT = np.linalg.svd(cnts[i].reshape(-1, 2), full_matrices=False)
    print(i, U, U.shape, S, VT)
    target.append(i)
    
  # 후보군에 대한 포인트 출력
  for i in target:
    ShowPoints(cnts[i], f"{i} Candidates")
  best_cnts = [cnts[i] for i in target]
  # print(best_cnts)
  # print(np.shape(cnts))
  # print(np.shape(best_cnts))
  index = LongestContourIndex(best_cnts)
  index = target[index]
  # print(index)
  
  # 출력하기
  ShowPoints(cnts[index], "Best Candidate")
  hull = cv2.convexHull(cnts[index])
  
  # hull_area = cv2.contourArea(hull)
  # cv2.imshow("Hull image", hull_area)
  # cv2.waitKey()
  
  # convex hull 포인트로 타원추정하기
  cv2.drawContours(imgs, [hull], 0, (255, 0, 0), 2)
  print(hull)
  elps_hull = cv2.fitEllipse(hull)
  cv2.ellipse(imgs, elps_hull, (0, 255, 0), 5)
  
  elps = cv2.fitEllipse(cnts[index])
  cv2.ellipse(imgs, elps, (0, 255, 0), 2)
  cv2.circle(img, (int(elps_hull[0][0]), int(elps_hull[0][1])), 10, (255, 255, 255), -1)
  
  # 게이지 영역만 잘라내기
  # => result_rgp
  mask = np.zeros_like(img_copy)
  mask = cv2.ellipse(mask, elps_hull, (255, 255, 255), -1)
  result = np.bitwise_and(img_copy, mask)
  result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  cv2.imshow("go", result_rgb)
  cv2.waitKey()
  
  
  
  
  # (xc,yc),(d1,d2),angle = elps_hull
  # rmajor = max(d1,d2)/2
  # if angle > 90:
  #     angle = angle - 90
  # else:
  #     angle = angle + 90
  # print(angle)
  # xtop = xc + math.cos(math.radians(angle))*rmajor
  # ytop = yc + math.sin(math.radians(angle))*rmajor
  # xbot = xc + math.cos(math.radians(angle+180))*rmajor
  # ybot = yc + math.sin(math.radians(angle+180))*rmajor
  # cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
  # cv2.imshow("Output image", imgs)
  # cv2.waitKey()
  
  # print(elps_hull)
  # new_img = np.zeros((h, w))
  # print(elps_hull[1][0]/2, elps_hull[1][1]/2)
  # for index_h in range(h):
  #   for index_w in range(w):
  #     if EllipseEq3(elps_hull[1][0]/2, elps_hull[1][1]/2, elps_hull[2], elps_hull[0][0], elps_hull[0][1], index_w, index_h) <= 1:
  #       new_img[index_h][index_w] = 255
  # cv2.imshow("go", new_img)
  # cv2.waitKey()
      
def EllipseFunc(elps):
  pass
def EllipseSkewCheck(elps):
  """ cv2.fitEllipse() 결과를 입력으로
  """
  if (abs(elps[1][0]/2 - elps[1][1]) / (elps[1][0]/2 + elps[1][1])) > 0.4:
    # 과도하게 휘어진 경우 => False
    return False
  return True
def EllipseSizeCheck(elps, h, w):
  """ 
  elps: cv2.fitEllipse() 결과
  h: 이미지 height
  w: 이미지 width
  """
  if ((elps[1][0] * elps[1][1]) / (h * w)) > 0.8 or ((elps[1][0] * elps[1][1]) / (h * w)) < 0.2:
    return False
  return True
def EllipseCenterDistanceCheck(elps, h, w, distance_thrsd=0.2):
  """
  elps: cv2.fitEllipse() 결과
  h: 이미지 height
  w: 이미지 width
  distance_thrsd: 퍼센티지
  """
  d = (h + w) / 2
  if np.sqrt((elps[0][0] - w/2) ** 2 + (elps[0][1] - h/2) ** 2) > (d * distance_thrsd):
    return False
  return True
def PointConvexHull():
  pass


if __name__=="__main__":
  # methods1()
  EllipseTest()
  # def EllipseEq2(long, short, angle, center_x, center_y, x, y):
  #   a = np.deg2rad(angle)
  #   x = x - center_x
  #   y = y - center_y
  #   p1 = ((long**2) * (math.sin(a) ** 2) + (short**2) * (math.cos(a) ** 2) * (x ** 2)) + 2 * (short ** 2 - long ** 2)* math.sin(a) * math.cos(a) * x * y + ((long ** 2) * (math.cos(a) ** 2) + (short**2) * (math.sin(a) ** 2)) * (y**2)
  #   p2 = (long**2) * (short**2)
  #   # print(p1, p2)
  #   return p1 >= p2
  
  
  
  # new_img = np.zeros((600, 600))
  # for index_h in range(600):
  #   for index_w in range(600):
  #     # if EllipseEq2(130, 193, 95, 264, 232, index_h, index_w) <= 1:
  #     #   pass
  #     if EllipseEq2(130, 200, 180, 300, 300, index_h, index_w):
  #       new_img[index_h][index_w] = 255
  # cv2.imshow("go", new_img)
  # cv2.waitKey()