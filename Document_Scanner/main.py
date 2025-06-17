import cv2
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import RCF
import os
import sys

#########################################################################
# 中文控制台乱码修复
if sys.platform.startswith('win'):
    import locale
    try:
        # 设置控制台编码为UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # 如果上面的方法不行，尝试设置locale
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except:
            pass

########################################################################
webCamFeed = True
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 640
widthImg = 480

# 图像路径列表
image_paths = [
    r'C:\Users\25068\Desktop\RCF_pytorch\Document_Scanner\test.jpg',
    r'C:\Users\25068\Desktop\RCF_pytorch\Document_Scanner\test_xiaoxi.jpg',
    r'C:\Users\25068\Desktop\RCF_pytorch\Document_Scanner\test_paper1.jpg',
    r'C:\Users\25068\Desktop\RCF_pytorch\Document_Scanner\test_wcc.jpg'
]

# 当前图像索引
current_path_index = 0

########################################################################

#############################################
#              model                        #
#############################################
# 1. 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = RCF()  # 先在CPU上创建模型
    
model_path = os.path.join(os.path.dirname(__file__), 'checkpoint_epoch7.pth')

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误：模型文件不存在 - {model_path}")
    sys.exit(1)

try:
    # 修复：添加map_location参数，确保模型可以在CPU上加载
    if device.type == 'cpu':
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load(model_path, weights_only=False)
    
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    
    # 在加载权重后再转移到目标设备
    model = model.to(device)
    model.eval()
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit(1)

# 2. 数据预处理 (与原论文dataset.py/BSDS_Dataset一致)
# 注意：这里的均值顺序是BGR，因为OpenCV默认读取BGR格式
mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

def rcf_edge_detection(img):
    """
    和原论文图像处理一致
    """
    # 预处理同上
    img = img.astype(np.float32)
    img -= mean
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    
    # 3. 准备输入张量
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # 添加batch维度
    img_tensor = img_tensor.to(device)
    
    # 4. 推理 (与test.py中的single_scale_test逻辑一致)
    with torch.no_grad():
        results = model(img_tensor)
        
        # 获取最后一个侧输出作为最终结果
        fuse_res = results[-1].detach().cpu().numpy()
        fuse_res = np.squeeze(fuse_res)  # 移除batch和channel维度
        
        # 反转并转换为uint8 (与test.py中的处理一致)
        # 注意：1 - 操作使边缘变黑，背景变白
        edge_map = ((1 - fuse_res) * 255).astype(np.uint8)

    return edge_map

def save_results(imgWarpColored, imgAdaptiveThre, current_index):
    """保存扫描结果"""
    try:
        # 创建保存目录
        save_dir = "scanned_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成文件名
        base_name = f"scan_{current_index + 1}"
        colored_path = os.path.join(save_dir, f"{base_name}_colored.jpg")
        final_path = os.path.join(save_dir, f"{base_name}_final.jpg")
        
        # 保存图像
        cv2.imwrite(colored_path, imgWarpColored)
        cv2.imwrite(final_path, imgAdaptiveThre)
        
        print("保存成功:")
        print(f"  彩色版本: {colored_path}")
        print(f"  最终版本: {final_path}")
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def print_controls():
    """打印控制说明"""
    print("\n" + "="*50)
    print("控制说明:")
    print("  'n' / '.' / '>' - 下一张图片")
    print("  'p' / ',' / '<' - 上一张图片")
    print("  's'            - 保存当前扫描结果")
    print("  'space'        - 暂停/继续")
    print("  'h'            - 显示帮助")
    print("  'q'            - 退出程序")
    print("  关闭窗口       - 退出程序")
    print("="*50)

# 检查图像文件是否存在
valid_paths = []
for i, path in enumerate(image_paths):
    if os.path.exists(path):
        valid_paths.append(path)
        print(f"图像 {i+1}: {os.path.basename(path)}")
    else:
        print(f"图像 {i+1} 不存在: {path}")

if not valid_paths:
    print("错误：没有找到任何有效的图像文件")
    sys.exit(1)

image_paths = valid_paths
current_path_index = 0

# 打印控制说明
print_controls()

#########################
#       主循环          #
#########################
biggest = None  # 初始化biggest变量

while True:
    #if webCamFeed:
        #success, img = cap.read()
    #else:
        #img = cv2.imread(current_path)

    # 获取当前图像路径
    current_path = image_paths[current_path_index]
    # 读取图像
    img = cv2.imread(current_path)
    if img is None:
        print(f"无法读取图像: {current_path}")
        break
        
    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    # RCF边缘检测
    try:
        img_rcf = rcf_edge_detection(img)
    except Exception as e:
        print(f"RCF边缘检测失败: {e}")
        break

    # 二值化 感觉RCF输出还是有点儿灰，估计训练的epoch不够，所以加了二值化
    img_binary = cv2.adaptiveThreshold(img_rcf, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 查找轮廓并绘制所有轮廓
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

    # 使用改进的轮廓检测函数查找最大轮廓
    biggest = utils.enhanced_biggest_contour(contours, widthImg, heightImg)
    
    imgBigContour = img.copy()
    
    if biggest.size != 0:
        # 确保biggest是正确的格式
        if len(biggest) == 4:
            biggest = utils.reorder(biggest)
            cv2.drawContours(imgBigContour, [biggest], -1, (0, 255, 0), 20)
            imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
            
            # 透视变换
            pts1 = np.float32(biggest.reshape(4, 2))
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            
            # 移除边缘像素
            imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
            
            # 应用自适应阈值
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
            
            imageArray = ([img, img_rcf, img_binary, imgContours],
                         [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
        else:
            print("检测到的轮廓不是四边形")
            imageArray = ([img, img_rcf, img_binary, imgContours],
                         [imgBlank, imgBlank, imgBlank, imgBlank])
    else:
        print("未检测到有效轮廓")
        imageArray = ([img, img_rcf, img_binary, imgContours],
                     [imgBlank, imgBlank, imgBlank, imgBlank])

    # 标签
    current_image_name = os.path.basename(current_path)
    labels = [["Original", "RCF Output", "Binary", "All Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utils.stackImages(imageArray, 0.75, labels)
    
    # 在图像上添加当前文件信息
    info_text = f"图像 {current_path_index + 1}/{len(image_paths)}: {current_image_name}"
    cv2.putText(stackedImage, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Document Scanner", stackedImage)

    # 按键检测
    key = cv2.waitKey(1) & 0xFF
    
    # 退出程序
    if key == ord('q'):
        print("按下 'q' 键，退出程序")
        break
    
    # 下一张图片 (支持多种按键)
    elif key == ord('n') or key == ord('.') or key == ord('>') or key == 83 or key == 3:  # 'n', '.', '>', 右箭头键
        current_path_index = (current_path_index + 1) % len(image_paths)
        next_image = os.path.basename(image_paths[current_path_index])
        print(f"切换到下一张: {next_image} ({current_path_index + 1}/{len(image_paths)})")
    
    # 上一张图片 (支持多种按键)
    elif key == ord('p') or key == ord(',') or key == ord('<') or key == 81 or key == 2:  # 'p', ',', '<', 左箭头键
        current_path_index = (current_path_index - 1) % len(image_paths)
        prev_image = os.path.basename(image_paths[current_path_index])
        print(f"切换到上一张: {prev_image} ({current_path_index + 1}/{len(image_paths)})")
    
    # 保存结果
    elif key == ord('s'):
        if biggest is not None and biggest.size != 0 and len(biggest) == 4:
            print("正在保存扫描结果...")
            if save_results(imgWarpColored, imgAdaptiveThre, current_path_index):
                print("保存完成")
            else:
                print("保存失败")
        else:
            print("没有检测到有效的四边形，无法保存")
    
    # 暂停功能
    elif key == ord(' '):
        print("程序暂停，按任意键继续...")
        cv2.waitKey(0)
        print("程序继续")
    
    # 帮助信息
    elif key == ord('h'):
        print_controls()
    
    # 检查窗口是否被关闭
    try:
        if cv2.getWindowProperty("Document Scanner", cv2.WND_PROP_VISIBLE) < 1:
            print("窗口被关闭，退出程序")
            break
    except:
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
print("程序结束")