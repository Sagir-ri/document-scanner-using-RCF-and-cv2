import cv2
import numpy as np

def stackImages(imgArray, scale, lables=[]):
    """堆叠图像显示函数"""
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: 
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: 
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                            (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                            (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                          cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def reorder(myPoints):
    """四边形顶点重新排序"""
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]  # top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # bottom-right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # bottom-left

    return myPointsNew

def drawRectangle(img, biggest, thickness):
    """绘制矩形边框"""
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img

def nothing(x):
    """trackbar回调函数"""
    pass

def initializeTrackbars(intialTracbarVals=0):
    """初始化trackbar"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 30, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 80, 255, nothing)

def valTrackbars():
    """获取trackbar值"""
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return Threshold1, Threshold2

def is_contour_on_border(contour, img_width, img_height, threshold):
    """检查轮廓是否过于接近图像边缘"""
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 检查边界框是否过于接近图像边缘
    if (x <= threshold or y <= threshold or 
        x + w >= img_width - threshold or 
        y + h >= img_height - threshold):
        return True
    
    # 检查轮廓点是否过于接近边缘
    for point in contour:
        px, py = point[0]
        if (px <= threshold or py <= threshold or 
            px >= img_width - threshold or 
            py >= img_height - threshold):
            return True
    
    return False

def is_reasonable_shape(contour, min_vertices=4, max_vertices=20):
    """检查轮廓形状是否合理（用于文档检测）"""
    # 使用多边形逼近检查顶点数量
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 文档通常是4边形，但允许一定的灵活性
    if len(approx) < min_vertices or len(approx) > max_vertices:
        return False
    
    # 检查轮廓的凸性（文档轮廓通常是凸的）
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    
    # 如果轮廓面积与凸包面积差异太大，可能不是文档
    if hull_area > 0:
        convexity_ratio = contour_area / hull_area
        if convexity_ratio < 0.7:
            return False
    
    return True

def enhanced_biggest_contour(contours, img_width, img_height):
    """
    增强版的最大轮廓查找函数，避免选择图片外沿
    专门优化用于文档检测
    """
    if len(contours) == 0:
        return np.array([])
    
    total_area = img_width * img_height
    min_area = total_area * 0.01  # 最小面积为图像的1%
    max_area = total_area * 0.85  # 最大面积为图像的85%，避免选择整个图像
    border_threshold = 15  # 边界阈值
    
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 面积过滤
        if area < min_area or area > max_area:
            continue
            
        # 边缘过滤 - 避免选择贴近边缘的轮廓
        if is_contour_on_border(contour, img_width, img_height, border_threshold):
            continue
            
        # 形状过滤 - 确保是合理的形状
        if not is_reasonable_shape(contour):
            continue
            
        # 多边形近似
        peri = cv2.arcLength(contour, True)
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果近似后顶点太多，尝试更严格的近似
        if len(approx) > 6:
            epsilon = 0.03 * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果仍然顶点太多，使用最小外接矩形
        if len(approx) > 6:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = np.int32(box).reshape(-1, 1, 2)
        
        # 确保至少有4个顶点
        if len(approx) >= 4:
            # 计算质量评分
            rect_area = cv2.contourArea(approx)
            
            # 计算长宽比
            rect = cv2.minAreaRect(approx)
            w, h = rect[1]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            
            # 计算紧凑度
            compactness = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            
            # 计算匹配度
            match_score = rect_area / area if area > 0 else 0
            
            # 质量控制
            if (compactness > 0.1 and 
                match_score > 0.6 and 
                aspect_ratio < 8 and
                len(approx) <= 8):  # 限制顶点数量
                
                # 如果是4个顶点，直接使用
                if len(approx) == 4:
                    final_approx = approx
                else:
                    # 如果不是4个顶点，转换为最小外接矩形
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    final_approx = np.int32(box).reshape(-1, 1, 2)
                
                candidates.append({
                    'contour': final_approx,
                    'area': area,
                    'score': area * compactness * match_score,
                    'compactness': compactness,
                    'match_score': match_score,
                    'aspect_ratio': aspect_ratio
                })
    
    # 选择最佳候选
    if candidates:
        # 按评分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回评分最高的候选
        best_candidate = candidates[0]
        return best_candidate['contour']
    
    return np.array([])