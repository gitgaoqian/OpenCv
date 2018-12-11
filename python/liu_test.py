# -*- coding: utf-8 -*-
import cv2
import numpy as np
#import cv2.cv as cv
import math
np.set_printoptions(threshold='nan')
levels = 3
e1 = cv2.getTickCount()

def PyDownMean(img):
    rows, cols = img.shape[:2]
    A = np.int(rows * 0.5)
    B = np.int(cols * 0.5)
    PyDownMean_gray = np.zeros((A, B), dtype=np.float)
    for m in range(A):
        for n in range(B):
            PyDownMean_gray[m, n] = 0.25 * img[2 * m, 2 * n] + 0.25 * img[2 * m + 1, 2 * n] + 0.25 * img[2 * m, 2 * n + 1] + 0.25 * img[2 * m + 1, 2 * n + 1]
    return PyDownMean_gray

def PyDownMeanN(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grayPyramidA = [img_gray.copy()]
    for i in range(1, levels):
        new_img = PyDownMean(img_gray)
        img_grayPyramidA.append(new_img)
        img_gray = new_img.copy()
    return img_grayPyramidA

def MAX(a, b):
    if (a > b):
        MAX = a
    else:
        MAX = b
    return MAX

def MIN(a, b):
    if (a > b):
        MIN = b
    else:
        MIN = a
    return MIN

def FindContours(img):
    Contours = []
    CentrCoord = []
    for i in range(levels):
        img[i] = img[i].astype(np.uint8)
        ret, binary = cv2.threshold(img[i], 200, 255, cv2.THRESH_BINARY_INV)
        (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rows, cols = img[i].shape[:2]
        """按中心旋转"""
        cX = 0.5 * cols
        cY = 0.5 * rows
        centrcoord = (cX, cY)
        Contours_len = len(contours)
        Contour = contours[0]
        if (Contours_len == 1):
            Contour = contours[0]
        else:
            for j in range(1, Contours_len):
                Contour = np.concatenate((Contour, contours[j]))
        Contours.append(Contour)
        CentrCoord.append(centrcoord)
    return Contours, CentrCoord

def ModelSurfaceFit(img, contours):
    invA = np.mat([[0.500000000,             0,             0,             -1,              0,  0.166670000],
                   [          0,   0.500000000,             0,              0,             -1,  0.166670000],
                   [          0,             0,   0.250000000,    -0.250000000,  -0.250000000,  0.250000000],
                   [         -1,             0,  -0.250000000,     2.416667000,   0.250000000, -0.750000000],
                   [          0,            -1,  -0.250000000,     0.250000000,   2.416667000, -0.750000000],
                   [0.166667000,   0.166667000,   0.250000000,    -0.750000000,  -0.750000000,  0.805556000]])
    localmat = np.zeros((3, 3), dtype=np.float)
    rows, cols = img.shape[:2]
    img_gx = np.zeros((rows, cols), dtype=np.float32)
    img_gy = np.zeros((rows, cols), dtype=np.float32)
    contours_x = np.array(contours[:, 0][:, 0])
    contours_y = np.array(contours[:, 0][:, 1])
    for i in range(len(contours_x)):
        localmat[0, 0] = np.float(img[contours_y[i]-1, contours_x[i]-1])
        localmat[0, 1] = np.float(img[contours_y[i]-1,   contours_x[i]])
        localmat[0, 2] = np.float(img[contours_y[i]-1, contours_x[i]+1])
        localmat[1, 0] = np.float(img[contours_y[i],   contours_x[i]-1])
        localmat[1, 1] = np.float(img[contours_y[i],     contours_x[i]])
        localmat[1, 2] = np.float(img[contours_y[i],   contours_x[i]+1])
        localmat[2, 0] = np.float(img[contours_y[i]+1, contours_x[i]-1])
        localmat[2, 1] = np.float(img[contours_y[i]+1,   contours_x[i]])
        localmat[2, 2] = np.float(img[contours_y[i]+1, contours_x[i]+1])
        h1 = localmat[0, 1] + localmat[1, 1] + localmat[2, 1] + 4 * (localmat[0, 2] + localmat[1, 2] + localmat[2, 2])
        h2 = localmat[1, 0] + localmat[1, 1] + localmat[1, 2] + 4 * (localmat[2, 0] + localmat[2][1] + localmat[2, 2])
        h3 = localmat[1, 1] + 2 * (localmat[2, 1] + localmat[1, 2]) + 4 * localmat[2, 2]
        h4 = localmat[0, 1] + localmat[1, 1] + localmat[2, 1] + 2 * (localmat[0, 2] + localmat[1, 2] + localmat[2, 2])
        h5 = localmat[1, 0] + localmat[1, 1] + localmat[1, 2] + 2 * (localmat[2, 0] + localmat[2, 1] + localmat[2, 2])
        h6 = localmat[0, 0] + localmat[1, 0] + localmat[2, 0] + localmat[0, 1] + localmat[1, 1] + localmat[2, 1] + localmat[0, 2] + localmat[1, 2] + localmat[2, 2]
        h = np.array([h1, h2, h3, h4, h5, h6])
        H = h.reshape(6, 1)
        K = invA*H
        gx = 2*K[0, 0] + K[2, 0] + K[3, 0]
        gy = 2*K[1, 0] + K[2, 0] + K[4, 0]
        s = math.sqrt(gy ** 2 + gx ** 2) + 0.000000000001
        img_gx[contours_y[i], contours_x[i]] = gx/s
        img_gy[contours_y[i], contours_x[i]] = gy/s

    return contours_x, contours_y, img_gx, img_gy

def GetModelGradient(img_gray, Contours):
    Gradient = []
    for i in range(levels):
        x, y, img_gx, img_gy = ModelSurfaceFit(img_gray[i], Contours[i])
        gradient = (x, y, img_gx, img_gy)
        Gradient.append(gradient)
    return Gradient

def TestSurfaceFit(img):
    invA = np.mat([[0.500000000,             0,             0,             -1,              0,  0.166670000],
                [          0,   0.500000000,             0,              0,             -1,  0.166670000],
                [          0,             0,   0.250000000,    -0.250000000,  -0.250000000,  0.250000000],
                [         -1,             0,  -0.250000000,     2.416667000,   0.250000000, -0.750000000],
                [          0,            -1,  -0.250000000,     0.250000000,   2.416667000, -0.750000000],
                [0.166667000,   0.166667000,   0.250000000,    -0.750000000,  -0.750000000,  0.805556000]])
    localmat = np.zeros((3, 3), dtype=np.float)
    rows, cols = img.shape[:2]
    img_gx = np.zeros((img.shape[0], img.shape[1]))
    img_gy = np.zeros((img.shape[0], img.shape[1]))
    s = np.zeros((img.shape[0], img.shape[1]))
    K1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    K2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    K3 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    K4 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    K5 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)

    for i in range(1, rows-1):
            for j in range(1, cols-1):
                #先定下3*3小区域
                localmat[0, 0] = np.float(img[i-1, j-1])
                localmat[0, 1] = np.float(img[i-1,   j])
                localmat[0, 2] = np.float(img[i-1, j+1])
                localmat[1, 0] = np.float(img[i,   j-1])
                localmat[1, 1] = np.float(img[i,     j])
                localmat[1, 2] = np.float(img[i,   j+1])
                localmat[2, 0] = np.float(img[i+1, j-1])
                localmat[2, 1] = np.float(img[i+1,   j])
                localmat[2, 2] = np.float(img[i+1, j+1])
                #定下H的值
                "正确"
                h1 = localmat[0, 1] + localmat[1, 1] + localmat[2, 1] + 4 * (localmat[0, 2] + localmat[1, 2] + localmat[2, 2])
                h2 = localmat[1, 0] + localmat[1, 1] + localmat[1, 2] + 4 * (localmat[2, 0] + localmat[2][1] + localmat[2, 2])
                h3 = localmat[1, 1] + 2 * (localmat[2, 1] + localmat[1, 2]) + 4 * localmat[2, 2]
                h4 = localmat[0, 1] + localmat[1, 1] + localmat[2, 1] + 2 * (localmat[0, 2] + localmat[1, 2] + localmat[2, 2])
                h5 = localmat[1, 0] + localmat[1, 1] + localmat[1, 2] + 2 * (localmat[2, 0] + localmat[2, 1] + localmat[2, 2])
                h6 = localmat[0, 0] + localmat[1, 0] + localmat[2, 0] + localmat[0, 1] + localmat[1, 1] + localmat[2, 1] + localmat[0, 2] + localmat[1, 2] + localmat[2, 2]
                h = np.array([h1, h2, h3, h4, h5, h6])
                H = h.reshape(6, 1)
                #根据AK=H求出K的值，也即是拟合的曲面方程的系数，得到系数后也就能得到每一点的梯度值
                K = invA*H
                K1[i, j] = K[0, 0]
                K2[i, j] = K[1, 0]
                K3[i, j] = K[2, 0]
                K4[i, j] = K[3, 0]
                K5[i, j] = K[4, 0]
                gx = 2*K1[i, j] + K3[i, j] + K4[i, j]
                gy = 2*K2[i, j] + K3[i, j] + K5[i, j]
                s[i, j] = math.sqrt(gy ** 2 + gx ** 2) + 0.000000000001
                img_gx[i, j] = gx / s[i, j]
                img_gy[i, j] = gy / s[i, j]
    return img_gx, img_gy, K1, K2, K3, K4, K5

def GetTestGradient(img_gray):
    Gradient = []
    for i in range(levels):
        img_gx, img_gy, K1, K2, K3, K4, K5 = TestSurfaceFit(img_gray[i])
        gradient = (img_gx, img_gy, K1, K2, K3, K4, K5)
        Gradient.append(gradient)
    return Gradient

# 求的是模板每一个轮廓点对应的目标图片上的点的亚像素精度的梯度值
def GetTestSubPixGradient( model_contours_x, model_contours_y, suitable_x, suitable_y, test_K1, test_K2, test_K3, test_K4, test_K5):
    t = np.ones(len(model_contours_x), dtype=np.int)
    T_Delta_GX = []
    T_Delta_GY = []
    k1 = test_K1[model_contours_y+suitable_y*t, model_contours_x+suitable_x*t]
    k2 = test_K2[model_contours_y+suitable_y*t, model_contours_x+suitable_x*t]
    k3 = test_K3[model_contours_y+suitable_y*t, model_contours_x+suitable_x*t]
    k4 = test_K4[model_contours_y+suitable_y*t, model_contours_x+suitable_x*t]
    k5 = test_K5[model_contours_y+suitable_y*t, model_contours_x+suitable_x*t]
    for delta_y in range(-10, 11):
        for delta_x in range(-10, 11):
            gx = 2*k1*(1+(delta_x/10.0))*t+(1+(delta_y/10.0))*k3*t+k4*t
            gy = 2*k2*(1+(delta_y/10.0))*t+(1+(delta_x/10.0))*k3*t+k5*t
            for i in range(len(gx)):
                s = math.sqrt(gy[i] ** 2 + gx[i] ** 2) + 0.000000000001
                gx[i] = gx[i]/s
                gy[i] = gy[i]/s

            T_Delta_GX.append(gx)
            T_Delta_GY.append(gy)
    return T_Delta_GX, T_Delta_GY

def FindLocalMaximum2D(S, threshold):
    Points = []
    point = []
    for i in range(1, S.shape[0] - 1):
        for j in range(1, S.shape[1] - 1):
            if (S[i, j] >= threshold):
                if (S[i, j] > S[i - 1, j - 1]
                    and S[i, j] > S[i - 1, j]
                    and S[i, j] > S[i - 1, j + 1]
                    and S[i, j] > S[i, j - 1]
                    and S[i, j] > S[i, j + 1]
                    and S[i, j] > S[i + 1, j - 1]
                    and S[i, j] > S[i + 1, j]
                    and S[i, j] > S[i + 1, j + 1]):
                    point = [j, i]
                    Points.append(point)
    return Points

def Atan2(a,b):
    if (a > 0.0 ):
        c = np.arctan(b / a)
    if (a < 0 and b >= 0):
        c = np.arctan(b / a) - cv.CV_PI
    if (a < 0 and b < 0):
        c = np.arctan(b / a) + cv.CV_PI
    if (a == 0.0 and b < 0.0):
        c = cv.CV_PI / 2
    if (a == 0.0 and b > 0.0):
        c = -cv.CV_PI / 2
    if (a == 0.0 and b == 0.0):
        c = math.atan2(b, a)  # 计算未旋转前每一个点的梯度夹角
    return c

def GetSubPixelCoordinate(index_max, suitable_x, suitable_y):
    rem = (index_max + 1) % 21
    quotient = (index_max + 1) / 21
    if (rem == 0):
        x = 2
        y = (quotient - 1) * 0.1
    else:
        x = (rem - 1) * 0.1
        y = quotient * 0.1
    new_x = suitable_x + x
    new_y = suitable_y + y
    return new_x, new_y

def GetModelRotatedPG(CentrCoord, Gradient):
    List = []
    for i in range(levels):
        m_x, m_y, m_gx, m_gy = Gradient[i][::]
        center_x, center_y = CentrCoord[i][::]
        Degree = []
        lenth = len(m_x)
        for delta in range(0, 360, 2**i):
            angle2rad = delta * cv2.cv.CV_PI / 180.0
            cosAngle = np.cos(angle2rad)
            sinAngle = np.sin(angle2rad)
            Rota_X = np.zeros(lenth, dtype=np.int64)#存储旋转后的横坐标
            Rota_Y = np.zeros(lenth, dtype=np.int64)#存储旋转后的纵坐标
            Rota_GX = np.zeros((m_gx.shape[0], m_gx.shape[1]), dtype=np.float64)#存储旋转后的gx
            Rota_GY = np.zeros((m_gx.shape[0], m_gx.shape[1]), dtype=np.float64)#存储旋转后的gy
            for j in range(lenth):#对每一个轮廓点进行旋转坐标以及梯度计算
                theta = math.atan2(m_gy[m_y[j], m_x[j]], m_gx[m_y[j], m_x[j]])
                gx = np.cos(theta + angle2rad)#计算旋转后每一个点的gx
                gy = np.sin(theta + angle2rad)#计算旋转后每一个点的gy
                x_offset = m_x[j] - center_x#计算x偏移量
                y_offset = m_y[j] - center_y#计算y偏移量
                new_x = x_offset * cosAngle - y_offset * sinAngle + center_x#计算旋转后的横坐标值
                new_y = x_offset * sinAngle + y_offset * cosAngle + center_y#计算旋转后的纵坐标值
                new_x = int(round(new_x))#将旋转后的横坐标值取整
                new_y = int(round(new_y))#将旋转后的纵坐标值取整
                Rota_X[j] = new_x #将旋转后的每一点的横坐标存起来
                Rota_Y[j] = new_y #将旋转后的每一点的纵坐标存起来
                Rota_GX[Rota_Y[j], Rota_X[j]] = gx #将旋转后的每一点的gx存起来
                Rota_GY[Rota_Y[j], Rota_X[j]] = gy #将旋转后的每一点的gy存起来
            degree = Rota_X, Rota_Y, Rota_GX[Rota_Y, Rota_X], Rota_GY[Rota_Y, Rota_X]
            Degree.append(degree)#存在对应层次的金字塔列表中
        List.append(Degree)
    return List

if __name__ == "__main__":
    model = cv2.imread("0330m.jpg")
    model_grayPyramidA = PyDownMeanN(model)
    MContours, MCentrCoord = FindContours(model_grayPyramidA)
    MGradient = GetModelGradient(model_grayPyramidA, MContours)
    M_Rota = GetModelRotatedPG(MCentrCoord, MGradient)
    test = cv2.imread("0330t.jpg")
    test_grayPyramidA = PyDownMeanN(test)
    TGradient = GetTestGradient(test_grayPyramidA)
    test0_gx, test0_gy, test0_K1, test0_K2, test0_K3, test0_K4, test0_K5 = TGradient[0][::]
    test1_gx, test1_gy, test1_K1, test1_K2, test1_K3, test1_K4, test1_K5 = TGradient[1][::]
    test2_gx, test2_gy, test2_K1, test2_K2, test2_K3, test2_K4, test2_K5 = TGradient[2][::]

    M0_Gradient = M_Rota[0]
    M1_Gradient = M_Rota[1]
    M2_Gradient = M_Rota[2]
    M0_rows, M0_cols = model_grayPyramidA[0].shape[:2]
    M1_rows, M1_cols = model_grayPyramidA[1].shape[:2]
    M2_rows, M2_cols = model_grayPyramidA[2].shape[:2]
    T0_rows, T0_cols = test_grayPyramidA[0].shape[:2]
    T1_rows, T1_cols = test_grayPyramidA[1].shape[:2]
    T2_rows, T2_cols = test_grayPyramidA[2].shape[:2]

    """最顶层金字塔进行粗匹配"""
    A2 = T2_rows - M2_rows + 1
    B2 = T2_cols - M2_cols + 1
    C2 = len(M2_Gradient[0][0])
    t2 = np.ones([C2], dtype=np.int)
    S2_Rough = np.zeros((A2, B2), dtype=np.float64)
    Angel2_Rough = np.zeros((A2, B2), dtype=np.int)
    grades_rough = np.zeros(90, dtype=np.float64)
    for i in range(A2):
        for j in range(B2):
            for delta in range(90):
                m_x_2 = M2_Gradient[delta][0]
                m_y_2 = M2_Gradient[delta][1]
                m_gx_2 = M2_Gradient[delta][2]
                m_gy_2 = M2_Gradient[delta][3]
                g2 = sum(m_gx_2 * test2_gx[m_y_2 + i * t2, m_x_2 + j * t2] + m_gy_2* test2_gy[m_y_2 + i * t2, m_x_2 + j * t2])/C2
                grades_rough[delta] = g2
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grades_rough)
            S2_Rough[i, j] = maxVal
            Angel2_Rough[i, j] = maxLoc[-1] * 4
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(S2_Rough)
    img = cv2.resize(S2_Rough, (640, 320))
    cv2.imshow("windows2", img)
    threshold = maxVal - 0.05
    pt = FindLocalMaximum2D(S2_Rough, threshold)
    for w in range(len(pt)):
        print ("最顶层金字塔粗匹配第%d个匹配位置%d，%d,旋转角度为%d,相似度量值为%f" % (w, pt[w][0], pt[w][1],Angel2_Rough[pt[w][1], pt[w][0]], S2_Rough[pt[w][1], pt[w][0]]))
        cv2.rectangle(test_grayPyramidA[2], (pt[w][0], pt[w][1]), (pt[w][0] + M2_cols, pt[w][1] + M2_rows),
                      (0, 100, 255), 2)
    cv2.imshow("Top_Rough", test_grayPyramidA[2])

    """最顶层金字塔进行精匹配"""
    print "最顶层金字塔进行精匹配"
    """分别在x,y附近两个像素范围类以0.1pix移动，计算相似度量值"""
    Match_Message_1 = []
    match_message_1 = ()
    for w in range(len(pt)):
        ang = Angel2_Rough[pt[w][1], pt[w][0]]
        MP_X_Rota = M_Rota[2][ang/4][0]
        MP_Y_Rota = M_Rota[2][ang/4][1]
        test_subpix_gx_2, test_subpix_gy_2 = GetTestSubPixGradient(MP_X_Rota, MP_Y_Rota, pt[w][0], pt[w][1], test2_K1,
                                                                  test2_K2, test2_K3, test2_K4, test2_K5)
        StartAng = MAX(ang/4 - 2, 0)
        EndAng = MIN(ang/4 + 2, 89)
        Grades_ = []  # 用来存441个位置处的最大分数
        Angle_ = []  # 用来存441个位置处的最佳旋转角度
        grades = np.zeros(90, dtype=np.float64)
        for i in range(441):
            Test_Point_GX = test_subpix_gx_2[i]
            Test_Point_GY = test_subpix_gy_2[i]
            for delta in range(StartAng, EndAng + 1):
                x = M_Rota[2][delta][0]
                y = M_Rota[2][delta][1]
                M_GX = M_Rota[2][delta][2]
                M_GY = M_Rota[2][delta][3]
                g = sum(M_GX * Test_Point_GX + M_GY * Test_Point_GY)
                grades[delta] = g / C2
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grades)
            Grades_.append(maxVal)
            Angle_.append(maxLoc[-1]*4)
        index_max = Grades_.index(max(Grades_))  # 返回最大值所在的下标
        """对索引进行判断"""
        max_point_x_2, max_point_y_2 = GetSubPixelCoordinate(index_max, pt[w][0], pt[w][1])
        Max_Angle = Angle_[index_max]
        print ("最顶层金字塔精匹配匹配第%d个匹配位置%.1f，%.1f匹配角度为%d,相似度量值为%f" % (w, max_point_x_2, max_point_y_2, Max_Angle, max(Grades_)))
        match_message_1 = (max_point_x_2, max_point_y_2, Max_Angle,max(Grades_))
        Match_Message_1.append(match_message_1)
        cv2.rectangle(test_grayPyramidA[2], (int(max_point_x_2), int(max_point_y_2)),
                      (int(max_point_x_2) + M2_cols, int(max_point_y_2) + M2_rows),
                      (0, 100, 255), 2)
    cv2.imshow("2", test_grayPyramidA[2])
    Match_Message_0 = []
    match_message_0 = ()
    "中间层金字塔进行粗匹配"
    print "中间层金字塔进行匹配"
    for m in range(len(Match_Message_1)):
        loc_sub_x_2, loc_sub_y_2 = Match_Message_1[m][:2]
        match_ang = Match_Message_1[m][-2]
        suitable_x_1 = np.int(loc_sub_x_2 * 2)
        suitable_y_1 = np.int(loc_sub_y_2 * 2)
        i0 = MAX(suitable_y_1 - 5, 0)
        i1 = MIN(suitable_y_1 + 5, T1_rows - M1_rows - 2)
        j0 = MAX(suitable_x_1 - 5, 0)
        j1 = MIN(suitable_x_1 + 5, T1_cols - M1_cols - 2)
        A1 = i1 - i0 + 1
        B1 = j1 - j0 + 1
        C1 = len(M_Rota[1][0][0])
        t1 = np.ones([C1], dtype=np.int)
        S1 = np.zeros((A1, B1), dtype=np.float64)
        StartAng = MAX(match_ang / 2 - 3, 0)
        EndAng = MIN(match_ang / 2 + 3, 179)
        grades1 = np.zeros((EndAng - StartAng + 1), dtype=np.float64)
        Angle1 = np.zeros((A1, B1), dtype=np.int)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for delta in range(StartAng, EndAng + 1):
                    x = M_Rota[1][delta][0]
                    y = M_Rota[1][delta][1]
                    M1_GX = M_Rota[1][delta][2]
                    M1_GY = M_Rota[1][delta][3]
                    g1 = sum(M1_GX * TGradient[1][0][y + i * t1, x + j * t1] + M1_GY * TGradient[1][1][y + i * t1, x + j * t1]) / C1
                    grades1[delta - StartAng] = g1
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grades1)
                Angle1[i - i0, j - j0] = (maxLoc[-1] + StartAng) * 2
                S1[i - i0, j - j0] = maxVal
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(S1)
        img = cv2.resize(S1, (640, 320))

        loc_x_1, loc_y_1 = maxLoc[::]
        ang_1 = Angle1[loc_y_1, loc_x_1]
        suitable_x_1 = loc_x_1 + j0
        suitable_y_1 = loc_y_1 + i0
        print ("中间层金字塔粗匹配第%d个匹配位置%d，%d匹配角度为%d,相似度量值为%f" % (m, suitable_x_1, suitable_y_1, ang_1, maxVal))
        cv2.rectangle(test_grayPyramidA[1], (suitable_x_1, suitable_y_1),
                      (suitable_x_1 + M1_cols, suitable_y_1 + M1_rows),
                      (0, 100, 255), 2)
        cv2.imshow("中间层粗匹配show", test_grayPyramidA[1])
        print "中间层金字塔进行亚像素级精匹配"
        "中间层金字塔进行亚像素级精匹配"
        """分别在x,y附近两个像素范围类以0.1pix移动，计算相似度量值"""
        MP_X_Rota_1 = M_Rota[1][ang_1 / 2][0]
        MP_Y_Rota_1 = M_Rota[1][ang_1 / 2][1]
        test_subpix_gx_1, test_subpix_gy_1 = GetTestSubPixGradient(MP_X_Rota_1, MP_Y_Rota_1, suitable_x_1, suitable_y_1,
                                                                  test1_K1, test1_K2, test1_K3, test1_K4, test1_K5)
        StartAng = MAX(ang_1 / 2 - 5, 0)
        EndAng = MIN(ang_1 / 2 + 5, 179)
        Grades1_ = []  # 用来存441个位置处的最大分数
        Angle1_ = []  # 用来存441个位置处的最佳旋转角度
        grades = np.zeros((EndAng - StartAng + 1), dtype=np.float64)
        for i in range(441):
            Test_Point_GX_1 = test_subpix_gx_1[i]
            Test_Point_GY_1 = test_subpix_gy_1[i]
            for delta in range(StartAng, EndAng + 1):
                x = M_Rota[1][delta][0]
                y = M_Rota[1][delta][1]
                M_GX = M_Rota[1][delta][2]
                M_GY = M_Rota[1][delta][3]
                g = sum(M_GX * Test_Point_GX_1 + M_GY * Test_Point_GY_1)
                grades[delta - StartAng] = g / C1
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grades)
            Grades1_.append(maxVal)
            Angle1_.append((maxLoc[-1] + StartAng) * 2)

        index1_max = Grades1_.index(max(Grades1_))  # 返回最大值所在的下标
        """对索引进行判断"""
        max_point_x_1, max_point_y_1 = GetSubPixelCoordinate(index1_max, suitable_x_1, suitable_y_1)
        Max_Angle = Angle1_[index1_max]
        cv2.rectangle(test_grayPyramidA[1], (int(max_point_x_1), int(max_point_y_1)),
                      (int(max_point_x_1) + M1_cols, int(max_point_y_1) + M1_rows),
                      (0, 100, 255), 2)
        match_message_0 = (max_point_x_1, max_point_y_1, Max_Angle, max(Grades1_))
        Match_Message_0.append(match_message_0)
        print ("中间层金字塔精匹配第%d个匹配位置%.1f，%.1f匹配角度为%d,相似度量值为%f" % (m, max_point_x_1, max_point_y_1, Max_Angle, max(Grades1_)))
    cv2.imshow("1", test_grayPyramidA[1])
    Match_Message = []
    match_message = ()
    "最底层金字塔进行粗匹配"
    print "最底层金字塔进行匹配"
    for n in range(len(Match_Message_0)):
        loc_sub_x_1, loc_sub_y_1 = Match_Message_0[n][:2]
        match_ang_1 = Match_Message_0[n][2]
        suitable_x_0 = np.int(loc_sub_x_1 * 2)
        suitable_y_0 = np.int(loc_sub_y_1 * 2)
        i0 = MAX(suitable_y_0 - 5, 0)
        i1 = MIN(suitable_y_0 + 5, T0_rows - M0_rows - 2)
        j0 = MAX(suitable_x_0 - 5, 0)
        j1 = MIN(suitable_x_0 + 5, T0_cols - M0_cols - 2)
        A0 = i1 - i0 + 1
        B0 = j1 - j0 + 1
        C0 = len(M_Rota[0][0][0])
        t0 = np.ones([C0], dtype=np.int)
        S0 = np.zeros((A0, B0), dtype=np.float64)
        StartAng = MAX(match_ang_1 - 5, 0)
        EndAng = MIN(match_ang_1 + 5, 359)
        grades0 = np.zeros((EndAng - StartAng + 1), dtype=np.float64)
        Angle0 = np.zeros((A0, B0), dtype=np.int)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for delta in range(StartAng, EndAng + 1):
                    x = M_Rota[0][delta][0]
                    y = M_Rota[0][delta][1]
                    M0_GX = M_Rota[0][delta][2]
                    M0_GY = M_Rota[0][delta][3]
                    g0 = sum(M0_GX * TGradient[0][0][y + i * t0, x + j * t0] + M0_GY * TGradient[0][1][
                        y + i * t0, x + j * t0]) / C0
                    grades0[delta - StartAng] = g0
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grades0)
                Angle0[i - i0, j - j0] = maxLoc[-1] + StartAng
                S0[i - i0, j - j0] = maxVal
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(S0)
        loc_x_0, loc_y_0 = maxLoc[::]
        ang_0 = Angle0[loc_y_0, loc_x_0]

        suitable_x_0 = loc_x_0 + j0
        suitable_y_0 = loc_y_0 + i0
        print ("最底层金字塔粗匹配第%d个匹配位置%d，%d匹配角度为%d,相似度量值为%f" % (n, suitable_x_0, suitable_y_0, ang_0, maxVal))
        cv2.rectangle(test_grayPyramidA[0], (suitable_x_0, suitable_y_0),
                      (suitable_x_0 + M0_cols, suitable_y_0 + M0_rows),
                      (0, 100, 255), 2)
        cv2.imshow("最底层粗匹配show", test_grayPyramidA[0])
        "最底层金字塔进行亚像素级精匹配"
        """分别在x,y附近两个像素范围类以0.1pix移动，计算相似度量值"""
        MP_X_Rota_0 = M_Rota[0][ang_0][0]
        MP_Y_Rota_0 = M_Rota[0][ang_0][1]
        test_subpix_gx_0, test_subpix_gy_0 = GetTestSubPixGradient(MP_X_Rota_0, MP_Y_Rota_0, suitable_x_0, suitable_y_0,
                                                                  test0_K1, test0_K2, test0_K3, test0_K4, test0_K5)
        StartAng = MAX(ang_0 - 5, 0)
        EndAng = MIN(ang_0 + 5, 359)
        Grades0_ = []  # 用来存441个位置处的最大分数
        Angle0_ = []  # 用来存441个位置处的最佳旋转角度
        grades = np.zeros((EndAng - StartAng + 1), dtype=np.float64)
        for i in range(441):
            Test_Point_GX_0 = test_subpix_gx_0[i]
            Test_Point_GY_0 = test_subpix_gy_0[i]
            for delta in range(StartAng, EndAng + 1):
                x = M_Rota[0][delta][0]
                y = M_Rota[0][delta][1]
                M_GX = M_Rota[0][delta][2]
                M_GY = M_Rota[0][delta][3]
                g = sum(M_GX * Test_Point_GX_0 + M_GY * Test_Point_GY_0)
                grades[delta - StartAng] = g / C0
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grades)
            Grades0_.append(maxVal)
            Angle0_.append(maxLoc[-1] + StartAng)
        index0_max = Grades0_.index(max(Grades0_))  # 返回最大值所在的下标
        """对索引进行判断"""
        max_point_x_0, max_point_y_0 = GetSubPixelCoordinate(index0_max, suitable_x_0, suitable_y_0)
        Max_Angle = Angle0_[index0_max]
        print ("最底层金字塔精匹配第%d个匹配位置%.1f，%.1f匹配角度为%d,相似度量值为%f" % (n, max_point_x_0, max_point_y_0, Max_Angle, max(Grades0_)))
        cv2.rectangle(test_grayPyramidA[0], (int(max_point_x_0), int(max_point_y_0)),
                      (int(max_point_x_0) + M0_cols, int(max_point_y_0) + M0_rows),
                      (0, 100, 255), 2)
        match_message = (max_point_x_0, max_point_y_0, Max_Angle, max(Grades0_))
        Match_Message.append(match_message)
    cv2.imshow("0", test_grayPyramidA[0])
    print Match_Message
    cv2.waitKey(0)



