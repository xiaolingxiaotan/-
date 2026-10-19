# cv2.calibrateCamera 源码解析

> 源码来源：[opencv/opencv · master · modules/calib3d/src/calibration.cpp](https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration.cpp)  
> 抓取版本：`OpenCV 4.15.0-dev`（master 与 `4.x` 分支该文件完全一致）  
> 文件规模：2991 行（其中与 `calibrateCamera` 直接相关的约 1900 行）  
> 抓取日期：2026-10-10


---

## 0. 先说结论：三件最容易被误导的事

1. **Python 的 `cv2.calibrateCamera` 在 C++ 里叫 `cv::calibrateCamera`**，但它在源码里只是个薄壳：两个公开重载全部转调 `cv::calibrateCameraRO`，真正干活的是 `calibrateCameraRO`。

2. **老教程里那个著名的 `cvCalibrateCamera2Internal` 已经不存在了。**现在的代码把标定引擎拆成了两个：`calibrateCameraInternalSchur`（默认）和 `calibrateCameraInternalBouguet`（回退）。如果你照着一篇 2020 年前后的博客去源码里搜 `cvCalibrateCamera2Internal`，会一无所获。

3. **默认引擎已经换成了 Schur 补稀疏 bundle adjustment。** 用 `CALIB_DISABLE_SCHUR_COMPLEMENT` 这个 flag 才能切回老的 Bouguet 引擎 —— 这条在 4.x 的官方文档里都有，但很多二手资料没跟上。


---

## 1. 文件与版本

| 项 | 值 |
| --- | --- |
| 源文件 | `modules/calib3d/src/calibration.cpp` |
| 公开声明 | `modules/calib3d/include/opencv2/calib3d.hpp`（`CV_EXPORTS_W`） |
| 常量定义 | `calib3d.hpp:605` `enum { CALIB_NINTRINSIC = 18, ... }` |
| 抓取版本 | master = `4.15.0-dev`，与 `4.x` 分支该文件逐字节相同 |
| 校验 | md5 `bf0451d13201f424b77f3145de4c988e` |

### 1.1 文件头的设计说明（原文）

```cpp
/*
    This program implements Levenberg-Marquardt optimization using Schur complement to calibrate cameras.
    It implements many speedups over the original implemntation.
    The techniques used here are primarily based upon the following papers:
    [1] "Bundle Adjustment -- A Modern Synthesis", Triggs et al., 1999
    [2] "SBA: A Software Package for Generic Sparse Bundle Adjustment", Lourakis and Argyros, 2000
*/
```


---

## 2. 调用链

```text
cv2.calibrateCamera(...)                          # Python 绑定
  └─ cv::calibrateCamera(...)                     # calibration.cpp:2609 / 2620 两个重载
       └─ cv::calibrateCameraRO(..., iFixedPoint = -1, ...)   # calibration.cpp:2650 【主函数】
            ├─ prepareCameraMatrix()               # 2571  内参输出规范化成 3x3
            ├─ prepareDistCoeffs()                 # 2594  畸变系数输出规范化成 5/12/14 维
            ├─ collectCalibrationData()            # 2452  点集打平成连续 Mat
            ├─ calibrateCameraInternalSchur()      # 1244  引擎 A（默认）
            │    ├─ initIntrinsicParams2D()        # 462   平面模板的内参初值
            │    └─ SchurLMSolver / JAccumulator    # 58 / 296   LM + Schur 补求解器
            └─ calibrateCameraInternalBouguet()    # 788   引擎 B（CALIB_DISABLE_SCHUR_COMPLEMENT）
                 ├─ initIntrinsicParams2D()        # 462
                 └─ LevMarq / subMatrixWithIndices # 570 / 543   经典 LM 求解器
```

注意第 2 行到第 3 行：`calibrateCamera` 本身不含任何算法逻辑，它只是把参数原样转发给 `calibrateCameraRO`，并把 `iFixedPoint` 固定为 `-1`。


---

## 3. 官方文档要点（`calib3d.hpp` `@brief` 摘录）

函数文档明确写了算法分四步：

1. **求内参初值**：仅平面标定板可用（未设 `CALIB_USE_INTRINSIC_GUESS` 时）。畸变系数初值全 0，除非指定了 `CALIB_FIX_K?`。

2. **求每幅视图的初始位姿**：把内参当作已知，逐视图调 `solvePnP` 那一套（`findExtrinsicCameraParams2`）。

3. **全局 Levenberg-Marquardt 优化**：最小化重投影误差，即所有 `imagePoints` 与「用当前参数投影 `objectPoints`」之间的平方距离总和（投影用 `projectPoints` 的同一套模型）。

4. **返回整体 RMS 重投影误差。**


关于引擎选择，文档原话：

> By default, the optimization follows a sparse bundle adjustment formulation with Schur complement; see [Triggs2000] and [Lourakis2009]. Use `CALIB_DISABLE_SCHUR_COMPLEMENT` to switch to the Bouguet calibration engine.


还有两条实用告警，做智能车标定时经常踩：

- 用非方阵棋盘时，`findChessboardCorners` 里如果写成 `patternSize=Size(rows, cols)`（应该是 `Size(cols, rows)`），`calibrateCamera` 会返回「畸变系数全 0、主点远离图像中心、fx/fy 相差 10 倍以上」这类看起来很荒谬但又不像报错的结果。

- 参数组合不支持或系统欠约束时，函数会直接抛异常，而不是返回垃圾值。


### 3.1 文档原文中的算法步骤段

```text
The algorithm performs the following steps:
  - Compute the initial intrinsic parameters (the option only available for planar
    calibration patterns) or read them from the input parameters.
  - Estimate the initial camera pose as if the intrinsic parameters have been already known.
    This is done using solvePnP.
  - Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection
    error, that is, the total sum of squared distances between the observed feature points
    imagePoints and the projected (using the current estimates for camera parameters and the
    poses) object points objectPoints.
```


---

## 4. 关键常量

| 符号 | 值 | 说明 |
| --- | --- | --- |
| `CALIB_NINTRINSIC` | `18` | 全局参数向量的内参维数，见 `calib3d.hpp:605` |
| `nimages` | `npoints.total()` | 视图（标定图片）数量 |
| `nparams` | `NINTRINSIC + nimages*6` | 不开 release-object 时的总参数数；每个视图 6 个外参（rvec 3 + tvec 3） |
| `nparams` | `+ maxPoints*3` | release-object 模式下再加上标定板三维点坐标 |

### 4.1 参数向量 `param` 的内存布局（全文最重要的约定）

```text
param[ 0.. 1]  fx, fy
param[ 2.. 3]  cx, cy
param[ 4.. 9]  k1 k2 p1 p2 k3 k4 k5 k6   ← 顺序由 mask 逐个开关
param[ ...  ]  s1 s2 s3 s4             (thin prism, 12 维模型)
param[16..17]  tauX tauY               (tilted sensor, 14 维模型)
--- 以上共 18 维，全部是全局参数 ---
param[NINTRINSIC + i*6 .. +5]           第 i 幅视图的 rvec(3) + tvec(3)
--- release-object 时追加 ---
param[NINTRINSIC + nimages*6 ..]        maxPoints 个三维点的 (x,y,z)
```

每个参数是否为未知量，由 `solver.mask`（Bouguet 引擎）或 `activeParams`（Schur 引擎）里的一个 `uchar` 控制，`0` = 固定，`1` = 参与优化。所有 `CALIB_FIX_*` flag 最终都翻译成对这张 mask 的写操作 —— 这就是为什么源码里会看到 `mask[0] = 0;` 这种看似魔数的写法。


---

## 5. 源码逐段

### 5.1 公开入口：`calibrateCamera` 与 `calibrateCameraRO`

**对应 `cv2.calibrateCamera` 的就是下面这两个 `calibrateCamera` 重载。**

`cv2.calibrateCamera` 是第二个重载（参数少），`cv2.calibrateCameraExtended` 是第一个（多返回内参/外参标准差和每视图误差）。两者都转调 `calibrateCameraRO`。

```cpp
double calibrateCamera( InputArrayOfArrays _objectPoints,
                        InputArrayOfArrays _imagePoints,
                        Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                        OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    return calibrateCamera(_objectPoints, _imagePoints, imageSize, _cameraMatrix, _distCoeffs,
                                         _rvecs, _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

double calibrateCamera(InputArrayOfArrays _objectPoints,
                       InputArrayOfArrays _imagePoints,
                       Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                       OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                       OutputArray stdDeviationsIntrinsics,
                       OutputArray stdDeviationsExtrinsics,
                       OutputArray _perViewErrors, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    return calibrateCameraRO(_objectPoints, _imagePoints, imageSize, -1, _cameraMatrix, _distCoeffs,
                             _rvecs, _tvecs, noArray(), stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                             noArray(), _perViewErrors, flags, criteria);
}
```


`calibrateCameraRO` 的 `iFixedPoint` 是模式开关：取 `[1, objectPoints[0].size()-2]` 走 release-object 方法（Strobl et al., ICCV 2011），否则走标准标定法。`calibrateCamera` 传 `-1`，所以永远是标准标定法。

```cpp
double calibrateCameraRO(InputArrayOfArrays _objectPoints,
                         InputArrayOfArrays _imagePoints,
                         Size imageSize, int iFixedPoint, InputOutputArray _cameraMatrix,
                         InputOutputArray _distCoeffs,
                         OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                         OutputArray newObjPoints,
                         int flags, TermCriteria criteria)
{
    CV_INSTRUMENT_REGION();

    return calibrateCameraRO(_objectPoints, _imagePoints, imageSize, iFixedPoint, _cameraMatrix,
                             _distCoeffs, _rvecs, _tvecs, newObjPoints, noArray(), noArray(),
                             noArray(), noArray(), flags, criteria);
}
```


下面是真正的主函数。请重点看它如何把「Python 侧的嵌套 list」变成「C++ 侧的连续 Mat」，以及如何按 flag 决定畸变系数是 5 / 12 / 14 维：

```cpp
double calibrateCameraRO(InputArrayOfArrays _objectPoints,
                         InputArrayOfArrays _imagePoints,
                         Size imageSize, int iFixedPoint, InputOutputArray _cameraMatrix,
                         InputOutputArray _distCoeffs,
                         OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                         OutputArray newObjPoints,
                         OutputArray stdDeviationsIntrinsics,
                         OutputArray stdDeviationsExtrinsics,
                         OutputArray stdDeviationsObjPoints,
                         OutputArray _perViewErrors, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    int rtype = CV_64F;

    CV_Assert( _cameraMatrix.needed() );
    CV_Assert( _distCoeffs.needed() );

    Mat cameraMatrix = _cameraMatrix.getMat();
    cameraMatrix = prepareCameraMatrix(cameraMatrix, rtype, flags);
    Mat distCoeffs = _distCoeffs.getMat();
    distCoeffs =
        (flags & CALIB_THIN_PRISM_MODEL) &&
        !(flags & CALIB_TILTED_MODEL) ?
            prepareDistCoeffs(distCoeffs, rtype, 12) :
            prepareDistCoeffs(distCoeffs, rtype, 14);
    if( !(flags & CALIB_RATIONAL_MODEL) &&
    (!(flags & CALIB_THIN_PRISM_MODEL)) &&
    (!(flags & CALIB_TILTED_MODEL)))
        distCoeffs = distCoeffs.rows == 1 ? distCoeffs.colRange(0, 5) : distCoeffs.rowRange(0, 5);

    int nimages = int(_objectPoints.total());
    CV_Assert( nimages > 0 );
    Mat objPt, imgPt, npoints, rvecM, tvecM, stdDeviationsM, errorsM;

    bool rvecs_needed = _rvecs.needed(), tvecs_needed = _tvecs.needed(),
            stddev_needed = stdDeviationsIntrinsics.needed(), errors_needed = _perViewErrors.needed(),
            stddev_ext_needed = stdDeviationsExtrinsics.needed();
    bool newobj_needed = newObjPoints.needed();
    bool stddev_obj_needed = stdDeviationsObjPoints.needed();

    bool rvecs_mat_vec = _rvecs.isMatVector();
    bool tvecs_mat_vec = _tvecs.isMatVector();

    if( rvecs_needed )
    {
        _rvecs.create(nimages, 1, CV_64FC3);

        if(rvecs_mat_vec)
            rvecM.create(nimages, 3, CV_64F);
        else
            rvecM = _rvecs.getMat();
    }

    if( tvecs_needed )
    {
        _tvecs.create(nimages, 1, CV_64FC3);

        if(tvecs_mat_vec)
            tvecM.create(nimages, 3, CV_64F);
        else
            tvecM = _tvecs.getMat();
    }

    collectCalibrationData( _objectPoints, _imagePoints, noArray(), iFixedPoint,
                            objPt, imgPt, noArray(), npoints );
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints.at<int>(0) - 1;

    newobj_needed = newobj_needed && releaseObject;
    int np = npoints.at<int>( 0 );
    Mat newObjPt;
    if( newobj_needed ) {
        newObjPoints.create( 1, np, CV_32FC3 );
        newObjPt = newObjPoints.getMat();
    }

    stddev_obj_needed = stddev_obj_needed && releaseObject;
    bool stddev_any_needed = stddev_needed || stddev_ext_needed || stddev_obj_needed;
    if( stddev_any_needed )
    {
        int sz = nimages*6 + CALIB_NINTRINSIC + (releaseObject ? np * 3 : 0);
        stdDeviationsM.create(sz, 1, CV_64F);
    }

    if( errors_needed )
    {
        _perViewErrors.create(nimages, 1, CV_64F);
        errorsM = _perViewErrors.getMat();
    }

    double reprojErr = 0;
    if( flags & CALIB_DISABLE_SCHUR_COMPLEMENT )
    {
        reprojErr = calibrateCameraInternalBouguet(
                objPt, imgPt, npoints, imageSize, iFixedPoint,
                cameraMatrix, distCoeffs,
                rvecM, tvecM,
                newObjPt,
                stdDeviationsM,
                errorsM, flags, criteria);
    }
    else
    {
        reprojErr = calibrateCameraInternalSchur(
                objPt, imgPt, npoints, imageSize, iFixedPoint,
                cameraMatrix, distCoeffs,
                rvecM, tvecM,
                newObjPt,
                stdDeviationsM,
                errorsM, flags, criteria);
    }

    if( stddev_needed )
    {
        stdDeviationsM.rowRange(0, CALIB_NINTRINSIC).copyTo(stdDeviationsIntrinsics);
    }
    if( stddev_ext_needed )
    {
        stdDeviationsM.rowRange(CALIB_NINTRINSIC, CALIB_NINTRINSIC + nimages*6).copyTo(stdDeviationsExtrinsics);
    }
    if( stddev_obj_needed )
    {
         int si = CALIB_NINTRINSIC + nimages*6;
         stdDeviationsM.rowRange(si, stdDeviationsM.rows).copyTo(stdDeviationsObjPoints);
    }

    cameraMatrix.copyTo(_cameraMatrix);
    distCoeffs.copyTo(_distCoeffs);

    for(int i = 0; i < nimages; i++ )
    {
        if( rvecs_needed && rvecs_mat_vec )
        {
            _rvecs.create(3, 1, CV_64F, i, true);
            Mat rv = _rvecs.getMat(i);
            Mat(rvecM.row(i).t()).copyTo(rv);
        }
        if( tvecs_needed && tvecs_mat_vec )
        {
            _tvecs.create(3, 1, CV_64F, i, true);
            Mat tv = _tvecs.getMat(i);
            Mat(tvecM.row(i).t()).copyTo(tv);
        }
    }
    return reprojErr;
}
```


### 5.2 输出容器规范化

`prepareCameraMatrix` 保证内参是 `3x3`；`prepareDistCoeffs` 按模型把畸变系数补齐到 5 / 12 / 14 维，并在输入逗号分隔的单行/单列之间统一方向。`initCameraMatrix2D` 是纯 Python 侧也会用到的「从单应矩阵估内参」接口，其内核就是 5.4 节的 `initIntrinsicParams2D`。

```cpp
static Mat prepareCameraMatrix(Mat& cameraMatrix0, int rtype, int flags)
{
    Mat cameraMatrix = Mat::eye(3, 3, rtype);
    if( cameraMatrix0.size() == cameraMatrix.size() )
        cameraMatrix0.convertTo(cameraMatrix, rtype);
    else if( flags & CALIB_USE_INTRINSIC_GUESS )
        CV_Error(Error::StsBadArg, "CALIB_USE_INTRINSIC_GUESS flag is set, but the camera matrix is not 3x3");
    return cameraMatrix;
}

Mat initCameraMatrix2D( InputArrayOfArrays objectPoints,
                        InputArrayOfArrays imagePoints,
                        Size imageSize, double aspectRatio )
{
    CV_INSTRUMENT_REGION();

    Mat objPt, imgPt, npoints, cameraMatrix;
    collectCalibrationData( objectPoints, imagePoints, noArray(),
                            objPt, imgPt, noArray(), npoints );
    initIntrinsicParams2D( objPt, imgPt, npoints, imageSize, cameraMatrix, aspectRatio );
    return cameraMatrix;
}

static Mat prepareDistCoeffs(Mat& distCoeffs0, int rtype, int outputSize)
{
    Size sz = distCoeffs0.size();
    int n = sz.area();
    if (n > 0)
        CV_Assert(sz.width == 1 || sz.height == 1);
    CV_Assert((int)distCoeffs0.total() <= outputSize);
    Mat distCoeffs = Mat::zeros(sz.width == 1 ? Size(1, outputSize) : Size(outputSize, 1), rtype);
    if( n ==  4 || n ==  5 || n ==  8 || n == 12 || n == 14 )
    {
        distCoeffs0.convertTo(distCoeffs(Rect(Point(), sz)), rtype);
    }
    return distCoeffs;
}
```


### 5.3 输入数据打平：`collectCalibrationData`

这是 Python list-of-list 跨过 pyopencv 边界后的第一站。它把 `vector<vector<Point3f>>` 之类的嵌套结构，打平成两个连续 `Mat`：`objPt/imgPt` 为 `1 x total` 的 `CV_32FC3 / CV_32FC2`，`npoints` 记录每幅视图的点数。后续所有引擎都只吃这三个东西。

```cpp
static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    int iFixedPoint,
                                    OutputArray objPt, OutputArray imgPt1, OutputArray imgPt2,
                                    OutputArray npoints )
{
    int nimages = (int)objectPoints.total();
    int total = 0;
    CV_Assert(nimages > 0);
    CV_CheckEQ(nimages, (int)imagePoints1.total(), "");
    if (!imagePoints2.empty())
    {
        CV_CheckEQ(nimages, (int)imagePoints2.total(), "");
        CV_Assert(imgPt2.needed());
    }

    for (int i = 0; i < nimages; i++)
    {
        Mat objectPoint = objectPoints.getMat(i);
        if (objectPoint.empty())
            CV_Error(cv::Error::StsBadSize, "objectPoints should not contain empty vector of vectors of points");
        int numberOfObjectPoints = objectPoint.checkVector(3, CV_32F, false);
        if (numberOfObjectPoints <= 0)
            CV_Error(cv::Error::StsUnsupportedFormat, "objectPoints should contain vector of vectors of points of type Point3f");

        Mat imagePoint1 = imagePoints1.getMat(i);
        if (imagePoint1.empty())
            CV_Error(cv::Error::StsBadSize, "imagePoints1 should not contain empty vector of vectors of points");
        int numberOfImagePoints = imagePoint1.checkVector(2, CV_32F, false);
        if (numberOfImagePoints <= 0)
            CV_Error(cv::Error::StsUnsupportedFormat, "imagePoints1 should contain vector of vectors of points of type Point2f");
        CV_CheckEQ(numberOfObjectPoints, numberOfImagePoints, "Number of object and image points must be equal");

        total += numberOfObjectPoints;
    }

    npoints.create(1, (int)nimages, CV_32S);
    objPt.create(1, (int)total, CV_32FC3);
    imgPt1.create(1, (int)total, CV_32FC2);
    Point2f* imgPtData2 = 0;

    Mat imgPt1Mat = imgPt1.getMat();
    if (!imagePoints2.empty())
    {
        imgPt2.create(1, (int)total, CV_32FC2);
        imgPtData2 = imgPt2.getMat().ptr<Point2f>();
    }

    Mat nPointsMat = npoints.getMat();
    Mat objPtMat = objPt.getMat();
    Point3f* objPtData = objPtMat.ptr<Point3f>();
    Point2f* imgPtData1 = imgPt1.getMat().ptr<Point2f>();

    for (int i = 0, j = 0; i < nimages; i++)
    {
        Mat objpt = objectPoints.getMat(i);
        Mat imgpt1 = imagePoints1.getMat(i);
        int numberOfObjectPoints = objpt.checkVector(3, CV_32F, false);
        if (!objpt.isContinuous())
            objpt = objpt.clone();
        if (!imgpt1.isContinuous())
            imgpt1 = imgpt1.clone();
        nPointsMat.at<int>(i) = numberOfObjectPoints;
        for (int n = 0; n < numberOfObjectPoints; ++n)
        {
            objPtData[j + n] = objpt.ptr<Point3f>()[n];
            imgPtData1[j + n] = imgpt1.ptr<Point2f>()[n];
        }

        if (imgPtData2)
        {
            Mat imgpt2 = imagePoints2.getMat(i);
            int numberOfImage2Points = imgpt2.checkVector(2, CV_32F, false);
            CV_CheckEQ(numberOfObjectPoints, numberOfImage2Points, "Number of object and image(2) points must be equal");
            if (!imgpt2.isContinuous())
                imgpt2 = imgpt2.clone();
            for (int n = 0; n < numberOfImage2Points; ++n)
            {
                imgPtData2[j + n] = imgpt2.ptr<Point2f>()[n];
            }
        }

        j += numberOfObjectPoints;
    }

    int ni = nPointsMat.at<int>(0);
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < ni - 1;
    // check object points. If not qualified, report errors.
    if( releaseObject )
    {
        for (int i = 1; i < nimages; i++)
        {
            if( nPointsMat.at<int>(i) != ni )
            {
                CV_Error( cv::Error::StsBadArg, "All objectPoints[i].size() should be equal when "
                                        "object-releasing method is requested." );
            }
            Mat ocmp = objPtMat.colRange(ni * i, ni * i + ni) != objPtMat.colRange(0, ni);
            ocmp = ocmp.reshape(1);
            if( countNonZero(ocmp) )
            {
                CV_Error( cv::Error::StsBadArg, "All objectPoints[i] should be identical when object-releasing"
                                        " method is requested." );
            }
        }
    }
}

static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    OutputArray objPt, OutputArray imgPtMat1, OutputArray imgPtMat2,
                                    OutputArray npoints )
{
    collectCalibrationData( objectPoints, imagePoints1, imagePoints2, -1, objPt, imgPtMat1,
                            imgPtMat2, npoints );
}
```


### 5.4 内参初值：`initIntrinsicParams2D`

平面标定板专用。对每幅视图的单应矩阵 `H` 施加两个约束（`h1`、`h2` 关于 `K^{-T}K^{-1}` 的正交归一约束），得到关于 `fx, fy, cx, cy` 的线性方程组，用 SVD 求最小二乘解。这正是 Zhang 氏标定法论文里「闭式解初值」那一步的实现。

```cpp
static void initIntrinsicParams2D( const Mat& objectPoints,
                         const Mat& imagePoints, const Mat& npoints,
                         Size imageSize, OutputArray cameraMatrix,
                         double aspectRatio )
{
    CV_Assert(npoints.type() == CV_32SC1 && (npoints.rows == 1 || npoints.cols == 1) && npoints.isContinuous());
    int nimages = npoints.rows + npoints.cols - 1;

    CV_Assert( objectPoints.type() == CV_32FC3 ||
               objectPoints.type() == CV_64FC3 );
    CV_Assert( imagePoints.type() == CV_32FC2 ||
               imagePoints.type() == CV_64FC2 );

    if( objectPoints.rows != 1 || imagePoints.rows != 1 )
        CV_Error( cv::Error::StsBadSize, "object points and image points must be a single-row matrices" );

    Mat_<double> matA(2*nimages, 2);
    Mat_<double> matb(2*nimages, 1, CV_64F );
    double fx, fy, cx, cy;
    cx = (!imageSize.width ) ? 0.5 : (imageSize.width - 1)*0.5;
    cy = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;

    // extract vanishing points in order to obtain initial value for the focal length
    int pos = 0;
    for(int i = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);
        Mat matM = objectPoints.colRange(pos, pos + ni);
        Mat _m = imagePoints.colRange(pos, pos + ni);
        pos += ni;

        Matx33d H;
        Mat matH0 = findHomography(matM, _m);
        CV_Assert(matH0.size() == Size(3, 3));
        matH0.convertTo(H, CV_64F);

        H(0, 0) -= H(2, 0)*cx; H(0, 1) -= H(2, 1)*cx; H(0, 2) -= H(2, 2)*cx;
        H(1, 0) -= H(2, 0)*cy; H(1, 1) -= H(2, 1)*cy; H(1, 2) -= H(2, 2)*cy;

        Vec3d h, v, d1, d2;
        Vec4d n;
        for(int j = 0; j < 3; j++ )
        {
            double t0 = H(j, 0), t1 = H(j, 1);
            h[j] = t0; v[j] = t1;
            d1[j] = (t0 + t1)*0.5;
            d2[j] = (t0 - t1)*0.5;
            n[0] += t0*t0; n[1] += t1*t1;
            n[2] += d1[j]*d1[j]; n[3] += d2[j]*d2[j];
        }

        for(int j = 0; j < 4; j++ )
            n[j] = 1./std::sqrt(n[j]);

        for(int j = 0; j < 3; j++ )
        {
            h[j] *= n[0]; v[j] *= n[1];
            d1[j] *= n[2]; d2[j] *= n[3];
        }

        matA(i*2+0, 0) = h[0]*v[0];   matA(i*2+0, 1) = h[1]*v[1];
        matA(i*2+1, 0) = d1[0]*d2[0]; matA(i*2+1, 1) = d1[1]*d2[1];
        matb(i*2+0) = -h[2]*v[2];
        matb(i*2+1) = -d1[2]*d2[2];
    }

    Vec2d f;
    solve(matA, matb, f, DECOMP_NORMAL + DECOMP_SVD);
    fx = std::sqrt(fabs(1./f[0]));
    fy = std::sqrt(fabs(1./f[1]));
    if( aspectRatio != 0 )
    {
        double tf = (fx + fy)/(aspectRatio + 1.);
        fx = aspectRatio*tf;
        fy = tf;
    }
    Mat(Matx33d(fx,  0, cx,
             0, fy, cy,
             0,  0,  1)).copyTo(cameraMatrix);
}
```


### 5.5 引擎 A（默认）：Schur 补稀疏 BA

参数维数在典型的 20 视图 × 50 点场景下是 `18 + 20*6 = 138`，雅可比稠密求解尚可；但 release-object 模式下还要加上 `50*3 = 150` 个三维点，参数逼近 300 维。Schur 补的作用是把「每个视图独有的 6 维位姿」当成局部块先消掉，只对全局参数（内参 + 共享三维点）解一个更小的方程组，再回代求位姿增量。


`SchurLMSolver` 的 `step()` 完整实现了这个过程：对 `V[i]` 加 `(1+lambda)` 正则后求逆 → 组装 Schur 补 `S = U - Σ Wi·Vi⁻¹·Wiᵀ` 与右端 `e = ea - Σ Wi·Vi⁻¹·ebi` → 按 mask 抽出活跃子块 → `cv::solve` → 回代出 `deltaLocal[i]`。`iterate()` 则是标准的 LM 接受/拒绝策略：误差上升就加大 lambda 并**不重算雅可比**，误差下降就接受并减小 lambda。

```cpp
class SchurLMSolver
{
public:
    enum State { STARTED, CALC_J, CHECK_ERR, DONE };

    SchurLMSolver()
        : currentError(0)
        , state(STARTED)
        , NINTRINSIC(18)
        , nimages(0)
        , n_global(0)
        , lambdaLg10(-3)
        , iters(0)
        , prevErrNorm(DBL_MAX)
        , solveMethod(cv::DECOMP_CHOLESKY)
    {
    }

    void init(int _nimages, int _NINTRINSIC, int nObjPoints,
              cv::TermCriteria _criteria)
    {
        nimages = _nimages;
        NINTRINSIC = _NINTRINSIC;
        n_global = NINTRINSIC + nObjPoints * 3;
        criteria = _criteria;

        U.create(n_global, n_global, CV_64F);
        ea.create(n_global, 1, CV_64F);
        deltaGlobal.create(n_global, 1, CV_64F);

        V.resize(nimages);
        V_inv.resize(nimages);
        W.resize(nimages);
        eb.resize(nimages);
        deltaLocal.resize(nimages);

        for (int i = 0; i < nimages; i++)
        {
            V[i].create(6, 6, CV_64F);
            V_inv[i].create(6, 6, CV_64F);
            W[i].create(n_global, 6, CV_64F);
            eb[i].create(6, 1, CV_64F);
            deltaLocal[i].create(6, 1, CV_64F);
        }

        state = STARTED;
        iters = 0;
        lambdaLg10 = -3;
        prevErrNorm = DBL_MAX;
        solveMethod = cv::DECOMP_CHOLESKY;
    }

    void reset()
    {
        U.setTo(0);
        ea.setTo(0);
        currentError = 0;
        for (int i = 0; i < nimages; i++)
        {
            V[i].setTo(0);
            W[i].setTo(0);
            eb[i].setTo(0);
        }
    }

    void step(const cv::Mat& mask)
    {
        const double LOG10 = std::log(10.0);
        double lambda = std::exp(lambdaLg10 * LOG10);

        // Invert V[i] blocks with regularization
        for (int i = 0; i < nimages; i++)
        {
            cv::Mat V_reg = V[i].clone();
            V_reg.diag() *= (1.0 + lambda);
            if (solveMethod == cv::DECOMP_QR)
            {
                cv::Mat I = cv::Mat::eye(6, 6, CV_64F);
                cv::solve(V_reg, I, V_inv[i], solveMethod);
            }
            else
            {
                cv::invert(V_reg, V_inv[i], solveMethod);
            }
        }

        // Form Schur complement S and reduced RHS e
        cv::Mat S = U.clone();
        S.diag() *= (1.0 + lambda);
        cv::Mat e = ea.clone();

        for (int i = 0; i < nimages; i++)
        {
            cv::Mat WtVinv = W[i] * V_inv[i];
            S -= WtVinv * W[i].t();
            e -= WtVinv * eb[i];
        }

        // Extract active submatrix using mask
        std::vector<uchar> mask_global(n_global);
        const uchar* mask_ptr = mask.ptr<uchar>();
        std::copy(mask_ptr, mask_ptr + NINTRINSIC, mask_global.begin());
        if (n_global > NINTRINSIC)
        {
            // Object points parameters follow intrinsic parameters in the global update vector
            std::copy(mask_ptr + NINTRINSIC + nimages * 6,
                      mask_ptr + NINTRINSIC + nimages * 6 + (n_global - NINTRINSIC),
                      mask_global.begin() + NINTRINSIC);
        }

        int nactive = 0;
        for (int i = 0; i < n_global; i++)
            if (mask_global[i]) nactive++;

        if (nactive == 0)
        {
            deltaGlobal.setTo(0);
            for (int i = 0; i < nimages; i++)
                deltaLocal[i].setTo(0);
            return;
        }

        cv::Mat S_sub(nactive, nactive, CV_64F);
        cv::Mat e_sub(nactive, 1, CV_64F);

        int ii = 0;
        for (int i = 0; i < n_global; i++)
        {
            if (!mask_global[i]) continue;
            e_sub.at<double>(ii) = e.at<double>(i);
            int jj = 0;
            for (int j = 0; j < n_global; j++)
            {
                if (!mask_global[j]) continue;
                S_sub.at<double>(ii, jj) = S.at<double>(i, j);
                jj++;
            }
            ii++;
        }

        // Solve the reduced system
        cv::Mat delta_sub;
        cv::solve(S_sub, e_sub, delta_sub, solveMethod);

        // Distribute the solution into the full update vector
        deltaGlobal.setTo(0);
        int j = 0;
        for (int i = 0; i < n_global; i++)
        {
            if (mask_global[i])
                deltaGlobal.at<double>(i) = delta_sub.at<double>(j++);
        }

        // Back-substitute to find the update for extrinsic parameters
        for (int i = 0; i < nimages; i++)
        {
            deltaLocal[i] = V_inv[i] * (eb[i] - W[i].t() * deltaGlobal);
        }
    }

    bool iterate(double errNorm, bool& needsJacobian)
    {
        needsJacobian = false;

        if (state == DONE)
        {
            return false;
        }

        if (state == STARTED)
        {
            state = CALC_J;
            needsJacobian = true;
            return true;
        }

        if (state == CALC_J)
        {
            prevErrNorm = errNorm;
            state = CHECK_ERR;
            return true;
        }

        // state == CHECK_ERR
        if (errNorm > prevErrNorm)
        {
            // Reject step
            if (++lambdaLg10 <= 16)
            {
                // Re-solve with higher lambda, DON'T recompute Jacobian
                return true;
            }
            // Lambda exhausted
        }

        // Accept step
        lambdaLg10 = std::max(lambdaLg10 - 1, -16);
        prevErrNorm = errNorm;

        if (++iters >= criteria.maxCount)
        {
            state = DONE;
            return false;
        }

        state = CALC_J;
        needsJacobian = true;
        return true;
    }

    cv::Mat getGlobalUpdate() const { return deltaGlobal; }
    cv::Mat getLocalUpdate(int i) const { return deltaLocal[i]; }
    void setSolveMethod(int method) { solveMethod = method; }

    cv::Mat U;
    std::vector<cv::Mat> V;
    std::vector<cv::Mat> W;
    cv::Mat ea;
    std::vector<cv::Mat> eb;
    double currentError;

    State state;
    int NINTRINSIC;
    int nimages;
    int n_global;
    int lambdaLg10;

private:
    int iters;
    double prevErrNorm;
    cv::TermCriteria criteria;
    int solveMethod;

    std::vector<cv::Mat> V_inv;
    cv::Mat deltaGlobal;
    std::vector<cv::Mat> deltaLocal;
};
```


`JAccumulator` 是雅可比累积器，继承 `ParallelLoopBody` 以多线程跑：

```cpp
class JAccumulator : public cv::ParallelLoopBody
{
public:
    JAccumulator(SchurLMSolver& _solver,
                 const cv::Mat& _matM, const cv::Mat& _m, const cv::Mat& _npoints,
                 const cv::Mat& _param, int _flags, double _aspectRatio,
                 int _NINTRINSIC, bool _releaseObject, int _maxPoints,
                 cv::Mutex& _globalMutex)
        : solver(_solver), matM(_matM), m(_m), npoints(_npoints),
          param(_param), flags(_flags), aspectRatio(_aspectRatio),
          NINTRINSIC(_NINTRINSIC), releaseObject(_releaseObject),
          maxPoints(_maxPoints), globalMutex(_globalMutex) {}

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int nimages_total = npoints.checkVector(1, CV_32S);
        int so = NINTRINSIC + nimages_total * 6;

        double fx = param.at<double>(0), fy = param.at<double>(1);
        double cx = param.at<double>(2), cy = param.at<double>(3);
        cv::Matx33d intrin(fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat dist = param.rowRange(4, 4 + 14);

        cv::Mat JiBuf(maxPoints * 2, NINTRINSIC, CV_64F);
        cv::Mat JeBuf(maxPoints * 2, 6, CV_64F);
        cv::Mat JoBuf;
        if (releaseObject)
            JoBuf.create(maxPoints * 2, maxPoints * 3, CV_64F);
        cv::Mat errBuf(maxPoints * 2, 1, CV_64F);

        int pos = 0;
        for (int i = 0; i < range.start; i++)
            pos += npoints.at<int>(i);

        double localErr = 0;

        for (int i = range.start; i < range.end; i++)
        {
            int ni = npoints.at<int>(i);
            int si = NINTRINSIC + i * 6;

            cv::Mat _ri = param.rowRange(si, si + 3);
            cv::Mat _ti = param.rowRange(si + 3, si + 6);

            cv::Mat _Mi = matM.colRange(pos, pos + ni);
            if (releaseObject)
            {
                _Mi = param.rowRange(so, so + ni * 3);
                _Mi = _Mi.reshape(3, 1);
            }
            cv::Mat _mi = m.colRange(pos, pos + ni);

            // Buffers for derivatives
            cv::Mat Ji = JiBuf.rowRange(0, ni * 2);
            Ji.setTo(0);
            cv::Mat Je = JeBuf.rowRange(0, ni * 2);
            Je.setTo(0);
            cv::Mat Jo;
            if (releaseObject)
            {
                Jo = JoBuf.rowRange(0, ni * 2).colRange(0, ni * 3);
                Jo.setTo(0);
            }
            cv::Mat err = errBuf.rowRange(0, ni * 2);
            cv::Mat _mp = err.reshape(2, 1);

            cv::Mat _dpdr, _dpdt, _dpdf, _dpdc, _dpdk, _dpdo;

            // Allocate contiguous buffers for Jacobians
            _dpdr.create(ni * 2, 3, CV_64F);
            _dpdt.create(ni * 2, 3, CV_64F);
            _dpdk.create(ni * 2, NINTRINSIC - 4, CV_64F);

            if (!(flags & cv::CALIB_FIX_FOCAL_LENGTH))
                _dpdf.create(ni * 2, 2, CV_64F);
            if (!(flags & cv::CALIB_FIX_PRINCIPAL_POINT))
                _dpdc.create(ni * 2, 2, CV_64F);
            if (releaseObject)
                _dpdo.create(ni * 2, ni * 3, CV_64F);

            cv::projectPoints(_Mi, _ri, _ti, intrin, dist, _mp,
                              _dpdr, _dpdt,
                              _dpdf.empty() ? cv::noArray() : _dpdf,
                              _dpdc.empty() ? cv::noArray() : _dpdc,
                              _dpdk,
                              _dpdo.empty() ? cv::noArray() : _dpdo,
                              (flags & cv::CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0.);

            // Copy Jacobians from projectPoints buffers to solver blocks
            _dpdr.copyTo(Je.colRange(0, 3));
            _dpdt.copyTo(Je.colRange(3, 6));

            if (!_dpdf.empty())
            {
                if (flags & cv::CALIB_FIX_ASPECT_RATIO)
                {
                    cv::Mat dpdf_fx = _dpdf.col(0);
                    cv::Mat dpdf_fy = _dpdf.col(1);
                    cv::scaleAdd(dpdf_fx, aspectRatio, dpdf_fy, dpdf_fy);
                }
                _dpdf.copyTo(Ji.colRange(0, 2));
            }
            if (!_dpdc.empty())
                _dpdc.copyTo(Ji.colRange(2, 4));

            _dpdk.copyTo(Ji.colRange(4, NINTRINSIC));

            if (!_dpdo.empty())
                _dpdo.copyTo(Jo);

            cv::subtract(_mp, _mi, _mp);
            localErr += cv::norm(err, cv::NORM_L2SQR);

            // Accumulate V and eb blocks (per-image)
            solver.V[i] = Je.t() * Je;
            solver.eb[i] = Je.t() * err;
            cv::Mat JitJe = Ji.t() * Je;
            JitJe.copyTo(solver.W[i].rowRange(0, NINTRINSIC));

            if (releaseObject)
            {
                cv::Mat JotJe = Jo.t() * Je;
                JotJe.copyTo(solver.W[i].rowRange(NINTRINSIC, NINTRINSIC + Jo.cols));
            }

            // Accumulate U and ea blocks (shared parameters)
            {
                cv::AutoLock lock(globalMutex);
                solver.U(cv::Rect(0, 0, NINTRINSIC, NINTRINSIC)) += Ji.t() * Ji;
                solver.ea.rowRange(0, NINTRINSIC) += Ji.t() * err;

                if (releaseObject)
                {
                    cv::Mat JitJo = Ji.t() * Jo;
                    solver.U(cv::Rect(NINTRINSIC, 0, Jo.cols, NINTRINSIC)) += JitJo;
                    solver.U(cv::Rect(0, NINTRINSIC, NINTRINSIC, Jo.cols)) += JitJo.t();
                    solver.U(cv::Rect(NINTRINSIC, NINTRINSIC, Jo.cols, Jo.cols)) += Jo.t() * Jo;
                    solver.ea.rowRange(NINTRINSIC, NINTRINSIC + Jo.cols) += Jo.t() * err;
                }

                solver.currentError += localErr;
            }

            pos += ni;
            localErr = 0;
        }
    }

private:
    SchurLMSolver& solver;
    const cv::Mat& matM;
    const cv::Mat& m;
    const cv::Mat& npoints;
    const cv::Mat& param;
    int flags;
    double aspectRatio;
    int NINTRINSIC;
    bool releaseObject;
    int maxPoints;
    cv::Mutex& globalMutex;
};
```


引擎主体：

```cpp
static double calibrateCameraInternalSchur( const Mat& objectPoints,
                                       const Mat& imagePoints, const Mat& npoints,
                                       Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs,
                                       Mat rvecs, Mat tvecs, Mat newObjPoints, Mat stdDevs,
                                       Mat perViewErr, int flags, const TermCriteria& termCrit )
{
    int NINTRINSIC = CALIB_NINTRINSIC;
    double reprojErr = 0;

    double aspectRatio = 0.;
    int nimages = npoints.checkVector(1, CV_32S);
    CV_Assert(nimages >= 1);
    int ndistCoeffs = (int)distCoeffs.total();
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints.at<int>(0) - 1;

    // 0. check the parameters & allocate buffers
    if (imageSize.width <= 0 || imageSize.height <= 0)
    {
        CV_Error(cv::Error::StsOutOfRange, "image width and height must be positive");
    }

    if (flags & CALIB_TILTED_MODEL)
    {
        // when the tilted sensor model is used the distortion coefficients matrix must have 14 parameters
        if (ndistCoeffs != 14)
        {
            CV_Error(cv::Error::StsBadArg, "The tilted sensor model must have 14 parameters");
        }
    }else
    {
        // when the thin prism model is used the distortion coefficients matrix must have 12 parameters
        if (flags & CALIB_THIN_PRISM_MODEL)
        {
            if (ndistCoeffs != 12)
            {
                CV_Error(cv::Error::StsBadArg, "Thin prism model must have 12 parameters");
            }
        }
    }

    if (!rvecs.empty())
    {
        int cn = rvecs.channels();
        CV_Assert(rvecs.depth() == CV_32F || rvecs.depth() == CV_64F);
        CV_Assert((rvecs.rows == nimages && (rvecs.cols * cn == 3)) ||
                  (rvecs.rows == 1 && rvecs.cols == nimages && cn == 3));
    }

    if (!tvecs.empty())
    {
        int cn = tvecs.channels();
        CV_Assert(tvecs.depth() == CV_32F || tvecs.depth() == CV_64F);
        CV_Assert((tvecs.rows == nimages && tvecs.cols * cn == 3) ||
                  (tvecs.rows == 1 && tvecs.cols == nimages && cn == 3));
    }

    CV_Assert(cameraMatrix.type() == CV_32F || cameraMatrix.type() == CV_64F);
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);

    CV_Assert(distCoeffs.type() == CV_32F || distCoeffs.type() == CV_64F);
    CV_Assert(distCoeffs.rows == 1 || distCoeffs.cols == 1);
    CV_Assert(ndistCoeffs == 4 || ndistCoeffs == 5 || ndistCoeffs == 8 ||
              ndistCoeffs == 12 || ndistCoeffs == 14);

    int total = 0, maxPoints = 0;
    for(int i = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);
        if( ni < 4 )
        {
            CV_Error_( cv::Error::StsOutOfRange, ("The number of points in the view #%d is < 4", i));
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    if( !newObjPoints.empty() )
    {
        int cn = newObjPoints.channels();
        CV_Assert(newObjPoints.depth() == CV_32F || newObjPoints.depth() == CV_64F);
        CV_Assert((newObjPoints.rows == maxPoints && newObjPoints.cols*cn == 3) ||
                  (newObjPoints.rows == 1 && newObjPoints.cols == maxPoints && cn == 3));
    }

    if( !stdDevs.empty() )
    {
        int cn = stdDevs.channels();
        CV_Assert(stdDevs.depth() == CV_32F || stdDevs.depth() == CV_64F);
        int nstddev = nimages*6 + NINTRINSIC + (releaseObject ? maxPoints*3 : 0);

        CV_Assert((stdDevs.rows == nstddev && stdDevs.cols*cn == 1) ||
                  (stdDevs.rows == 1 && stdDevs.cols == nstddev && cn == 1));
    }

    Mat matM( 1, total, CV_64FC3 );
    Mat _m( 1, total, CV_64FC2 );
    Mat allErrors(1, total, CV_64FC2);

    if (objectPoints.channels() == 3)
    {
        objectPoints.convertTo(matM, CV_64F);
    }else
    {
        convertPointsToHomogeneous(objectPoints, matM);
        matM.convertTo(matM, CV_64F);
    }

    if (imagePoints.channels() == 2)
    {
        imagePoints.convertTo(_m, CV_64F);
    }else
    {
        convertPointsFromHomogeneous(imagePoints, _m);
        _m.convertTo(_m, CV_64F);
    }

    int nparams = NINTRINSIC + nimages*6;
    if( releaseObject )
        nparams += maxPoints * 3;

    std::vector<double> k(14, 0.0);
    Mat _k( distCoeffs.rows, distCoeffs.cols, CV_64F, k.data());
    if( distCoeffs.total() < 8 )
    {
        if( distCoeffs.total() < 5 )
            flags |= CALIB_FIX_K3;
        flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
    }
    const double minValidAspectRatio = 0.01;
    const double maxValidAspectRatio = 100.0;

    Matx33d A;
    cameraMatrix.convertTo(A, CV_64F);

    // 1. initialize intrinsic parameters & LM solver
    if( flags & CALIB_USE_INTRINSIC_GUESS )
    {
        if( A(0, 0) <= 0 || A(1, 1) <= 0 )
            CV_Error( cv::Error::StsOutOfRange, "Focal length (fx and fy) must be positive" );
        if( A(0, 2) < 0 || A(0, 2) >= imageSize.width ||
            A(1, 2) < 0 || A(1, 2) >= imageSize.height )
            CV_Error( cv::Error::StsOutOfRange, "Principal point must be within the image" );
        if( fabs(A(0, 1)) > 1e-5 )
            CV_Error( cv::Error::StsOutOfRange, "Non-zero skew is not supported by the function" );
        if( fabs(A(1, 0)) > 1e-5 || fabs(A(2, 0)) > 1e-5 ||
            fabs(A(2, 1)) > 1e-5 || fabs(A(2,2)-1) > 1e-5 )
            CV_Error( cv::Error::StsOutOfRange,
                "The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape" );
        A(0, 1) = A(1, 0) = A(2, 0) = A(2, 1) = 0.;
        A(2, 2) = 1.;

        if (flags & CALIB_FIX_ASPECT_RATIO)
        {
            aspectRatio = A(0, 0) / A(1, 1);
            if (aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio)
            {
                CV_Error(cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect");
            }
        }
        distCoeffs.convertTo(_k, CV_64F);
    }else
    {
        Scalar mean, sdv;
        meanStdDev(matM, mean, sdv);
        if (fabs(mean[2]) > 1e-5 || fabs(sdv[2]) > 1e-5)
        {
            CV_Error(cv::Error::StsBadArg,
            "For non-planar calibration rigs the initial intrinsic matrix must be specified");
        }
        for (int i = 0; i < total; i++)
        {
            matM.at<Point3d>(i).z = 0.;
        }

        if (flags & CALIB_FIX_ASPECT_RATIO)
        {
            aspectRatio = A(0, 0);
            aspectRatio /= A(1, 1);
            if (aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio)
            {
                CV_Error(cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect");
            }
        }
        initIntrinsicParams2D(matM, _m, npoints, imageSize, A, aspectRatio);
    }

    SchurLMSolver solver;
    solver.init(nimages, NINTRINSIC, releaseObject ? maxPoints : 0, termCrit);
    if (flags & CALIB_USE_LU)
        solver.setSolveMethod(cv::DECOMP_LU);
    else if (flags & CALIB_USE_QR)
        solver.setSolveMethod(cv::DECOMP_QR);

    Mat_<double> param_m(nparams, 1);
    std::vector<uchar> mask_vec(nparams, 1);

    param_m(0) = A(0, 0); param_m(1) = A(1, 1); param_m(2) = A(0, 2); param_m(3) = A(1, 2);
    for (int i = 0; i < 14; i++)
        param_m(4 + i) = k[i];

    if (flags & CALIB_FIX_ASPECT_RATIO)
        mask_vec[0] = 0;
    if (flags & CALIB_FIX_FOCAL_LENGTH)
        mask_vec[0] = mask_vec[1] = 0;
    if (flags & CALIB_FIX_PRINCIPAL_POINT)
        mask_vec[2] = mask_vec[3] = 0;
    if (flags & CALIB_ZERO_TANGENT_DIST)
    {
        param_m(6) = param_m(7) = 0;
        mask_vec[6] = mask_vec[7] = 0;
    }
    if (!(flags & CALIB_RATIONAL_MODEL))
        flags |= CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6;
    if (!(flags & CALIB_THIN_PRISM_MODEL))
        flags |= CALIB_FIX_S1_S2_S3_S4;
    if (!(flags & CALIB_TILTED_MODEL))
        flags |= CALIB_FIX_TAUX_TAUY;

    mask_vec[4] = !(flags & CALIB_FIX_K1);
    mask_vec[5] = !(flags & CALIB_FIX_K2);
    if (flags & CALIB_FIX_TANGENT_DIST)
        mask_vec[6] = mask_vec[7] = 0;
    mask_vec[8] = !(flags & CALIB_FIX_K3);
    mask_vec[9] = !(flags & CALIB_FIX_K4);
    mask_vec[10] = !(flags & CALIB_FIX_K5);
    mask_vec[11] = !(flags & CALIB_FIX_K6);

    if (flags & CALIB_FIX_S1_S2_S3_S4)
    {
        mask_vec[12] = 0;
        mask_vec[13] = 0;
        mask_vec[14] = 0;
        mask_vec[15] = 0;
    }

    if (flags & CALIB_USE_INTRINSIC_GUESS)
    {
        CV_Assert(nparams == NINTRINSIC + nimages * 6);
    }

    if (flags & CALIB_FIX_ASPECT_RATIO)
    {
        mask_vec[0] = 0;
        if (aspectRatio <= 0)
            CV_Error(Error::StsBadArg, "CALIB_FIX_ASPECT_RATIO is set but aspectRatio is not positive");
    }

    if (flags & CALIB_FIX_TAUX_TAUY)
    {
        mask_vec[16] = mask_vec[17] = 0;
    }

    if (releaseObject)
    {
        int s = NINTRINSIC + nimages * 6;
        std::copy(matM.ptr<double>(), matM.ptr<double>(0, maxPoints - 1) + 3,
                  param_m.ptr<double>() + s);
        mask_vec[s + 0] = 0;
        mask_vec[s + 1] = 0;
        mask_vec[s + 2] = 0;
        mask_vec[s + iFixedPoint * 3 + 0] = 0;
        mask_vec[s + iFixedPoint * 3 + 1] = 0;
        mask_vec[s + iFixedPoint * 3 + 2] = 0;
        mask_vec[nparams - 1] = 0;
    }

    Mat mask(mask_vec);
    int nparams_nz = countNonZero(mask);

    if (nparams_nz >= 2 * total)
    {
        CV_Error_(Error::StsBadArg,
                  ("There should be less vars to optimize (having %d) than the "
                   "number of residuals (%d = 2 per point)", nparams_nz, 2 * total));
    }

    // Initialize extrinsic parameters
    for (int i = 0, pos = 0; i < nimages; i++)
    {
        int ni = npoints.at<int>(i);

        int s = NINTRINSIC + i*6;
        Mat _ri = param_m.rowRange(s, s + 3);
        Mat _ti = param_m.rowRange(s + 3, s + 6);

        Mat _Mi = matM.colRange(pos, pos + ni);
        Mat _mi = _m.colRange(pos, pos + ni);

        findExtrinsicCameraParams2(_Mi, _mi, Mat(A), _k, _ri, _ti, 0);

        pos += ni;
    }

    // 3. run the optimization
    Mat_<double> prev_param = param_m.clone();
    Mutex globalMutex;

    // Compute initial error
    Mat allErrorsBuf(1, total, CV_64FC2);
    reprojErr = 0;
    int so = NINTRINSIC + nimages * 6;
    int pos = 0;
    for (int i = 0; i < nimages; i++)
    {
        int ni = npoints.at<int>(i);
        int si = NINTRINSIC + i * 6;

        Mat _ri = param_m.rowRange(si, si + 3);
        Mat _ti = param_m.rowRange(si + 3, si + 6);

        Mat _Mi = matM.colRange(pos, pos + ni);
        if (releaseObject)
        {
            _Mi = param_m.rowRange(so, so + ni * 3);
            _Mi = _Mi.reshape(3, 1);
        }
        Mat _mi = _m.colRange(pos, pos + ni);
        Mat _me = allErrorsBuf.colRange(pos, pos + ni);

        double fx = param_m(0), fy = param_m(1), cx = param_m(2), cy = param_m(3);
        if (flags & CALIB_FIX_ASPECT_RATIO)
            fx = fy * aspectRatio;
        Matx33d intrin(fx, 0, cx, 0, fy, cy, 0, 0, 1);
        Mat dist = param_m.rowRange(4, 4 + 14);

        projectPoints(_Mi, _ri, _ti, intrin, dist, _me);
        subtract(_me, _mi, _me);

        reprojErr += norm(_me, NORM_L2SQR);
        pos += ni;
    }

    bool recomputeFinalErrors = (termCrit.maxCount == 0);
    bool jacobianAtCurrentParams = false;
    if (termCrit.maxCount > 0)
    {
        solver.reset();

        parallel_for_(Range(0, nimages),
                      JAccumulator(solver, matM, _m, npoints, param_m,
                                   flags, aspectRatio, NINTRINSIC,
                                   releaseObject, maxPoints, globalMutex));
        jacobianAtCurrentParams = true;
        // JAccumulator acc(solver, matM, _m, npoints, param_m,
        //                         flags, aspectRatio, NINTRINSIC,
        //                         releaseObject, maxPoints, globalMutex);
        // acc(Range(0, nimages));
        solver.step(mask);

        double prevErr = reprojErr;
        // Apply the initial step
        param_m.copyTo(prev_param);
        // 1. Global parameters (Intrinsics + Objects)
        for (int kk = 0; kk < NINTRINSIC; kk++)
            param_m(kk) -= solver.getGlobalUpdate().at<double>(kk);

        if (releaseObject)
        {
             int param_obj_start = NINTRINSIC + nimages * 6;
             int num_obj_params = solver.n_global - NINTRINSIC;
             for (int kk = 0; kk < num_obj_params; kk++)
                 param_m(param_obj_start + kk) -= solver.getGlobalUpdate().at<double>(NINTRINSIC + kk);
        }
        for (int i = 0; i < nimages; i++)
        {
            int si = NINTRINSIC + i * 6;
            for (int kk = 0; kk < 6; kk++)
                param_m(si + kk) -= solver.getLocalUpdate(i).at<double>(kk);
        }
        jacobianAtCurrentParams = false;
        for (int iter = 0; iter < termCrit.maxCount; iter++)
        {
            if (flags & CALIB_FIX_ASPECT_RATIO)
            {
                param_m(0) = param_m(1) * aspectRatio;
            }

            reprojErr = 0;
            int pos_iter = 0;
            for (int i = 0; i < nimages; i++)
            {
                int ni = npoints.at<int>(i);
                int si = NINTRINSIC + i * 6;

                Mat _ri = param_m.rowRange(si, si + 3);
                Mat _ti = param_m.rowRange(si + 3, si + 6);

                Mat _Mi = matM.colRange(pos_iter, pos_iter + ni);
                if (releaseObject)
                {
                    _Mi = param_m.rowRange(so, so + ni * 3).reshape(3, 1);
                }
                Mat _mi = _m.colRange(pos_iter, pos_iter + ni);
                Mat _me = allErrorsBuf.colRange(pos_iter, pos_iter + ni);

                double fx = param_m(0), fy = param_m(1), cx = param_m(2), cy = param_m(3);
                Matx33d intrin(fx, 0, cx, 0, fy, cy, 0, 0, 1);
                Mat dist = param_m.rowRange(4, 4 + 14);

                projectPoints(_Mi, _ri, _ti, intrin, dist, _me);
                subtract(_me, _mi, _me);

                double viewErr = norm(_me, NORM_L2SQR);
                if (!perViewErr.empty())
                {
                    perViewErr.at<double>(i) = std::sqrt(viewErr / ni);
                }
                reprojErr += viewErr;
                pos_iter += ni;
            }

            // Accept or reject the step
            if (reprojErr < prevErr)
            {
                // Step accepted
                solver.lambdaLg10 = std::max(solver.lambdaLg10 - 1, -16);
                prevErr = reprojErr;

                // Check convergence BEFORE saving new prev_param
                double paramChange = norm(param_m, prev_param, NORM_L2) /
                                     (norm(prev_param, NORM_L2) + DBL_EPSILON);
                param_m.copyTo(prev_param);
                if (paramChange < termCrit.epsilon)
                {
                    break;
                }

                // Compute new Jacobian at new position
                solver.reset();
                parallel_for_(Range(0, nimages),
                              JAccumulator(solver, matM, _m, npoints, param_m,
                                           flags, aspectRatio, NINTRINSIC,
                                           releaseObject, maxPoints, globalMutex));
                jacobianAtCurrentParams = true;
            }else
            {
                // Step rejected, increase lambda and recompute step
                if (++solver.lambdaLg10 > 16)
                {
                    // Lambda exhausted, restore best params
                    prev_param.copyTo(param_m);
                    recomputeFinalErrors = true;
                    break;
                }

                // Revert params
                prev_param.copyTo(param_m);
            }

            // Compute step (possibly with new lambda)
            solver.step(mask);

            // Apply step. Update layouts:
            // deltaGlobal: [Intrinsics (0..NINTRINSIC-1) | Object points (NINTRINSIC..n_global-1)]
            // param_m: [Intrinsics | Extrinsic parameters | Object points]

            // Apply Intrinsics update
            for (int kk = 0; kk < NINTRINSIC; kk++)
            {
                param_m(kk) -= solver.getGlobalUpdate().at<double>(kk);
            }

            // Apply Objects update (if any)
            if (releaseObject)
            {
                 int param_obj_start = NINTRINSIC + nimages * 6;
                 int num_obj_params = solver.n_global - NINTRINSIC;
                 for (int kk = 0; kk < num_obj_params; kk++)
                 {
                     param_m(param_obj_start + kk) -= solver.getGlobalUpdate().at<double>(NINTRINSIC + kk);
                 }
            }

            // Apply Local parameters (Extrinsics)
            for (int i = 0; i < nimages; i++)
            {
                int si = NINTRINSIC + i * 6;
                for (int kk = 0; kk < 6; kk++)
                    param_m(si + kk) -= solver.getLocalUpdate(i).at<double>(kk);
            }
            jacobianAtCurrentParams = false;
        }
    }

    if (flags & CALIB_FIX_ASPECT_RATIO)
    {
        param_m(0) = param_m(1) * aspectRatio;
    }

    if (recomputeFinalErrors)
    {
        Mat finalErrorsBuf;
        Mat* errorsPtr = &allErrorsBuf;
        if (!stdDevs.empty())
        {
            finalErrorsBuf.create(1, total, CV_64FC2);
            errorsPtr = &finalErrorsBuf;
        }

        Mat& errors = *errorsPtr;
        reprojErr = 0;
        pos = 0;
        for (int i = 0; i < nimages; i++)
        {
            int ni = npoints.at<int>(i);
            int si = NINTRINSIC + i * 6;

            Mat _ri = param_m.rowRange(si, si + 3);
            Mat _ti = param_m.rowRange(si + 3, si + 6);

            Mat _Mi = matM.colRange(pos, pos + ni);
            if (releaseObject)
            {
                _Mi = param_m.rowRange(so, so + ni * 3).reshape(3, 1);
            }
            Mat _mi = _m.colRange(pos, pos + ni);
            Mat _me = errors.colRange(pos, pos + ni);

            double fx = param_m(0), fy = param_m(1), cx = param_m(2), cy = param_m(3);
            Matx33d intrin(fx, 0, cx, 0, fy, cy, 0, 0, 1);
            Mat dist = param_m.rowRange(4, 4 + 14);

            projectPoints(_Mi, _ri, _ti, intrin, dist, _me);
            subtract(_me, _mi, _me);

            double viewErr = norm(_me, NORM_L2SQR);
            if (!perViewErr.empty())
            {
                perViewErr.at<double>(i) = std::sqrt(viewErr / ni);
            }
            reprojErr += viewErr;
            pos += ni;
        }
    }

    if (!stdDevs.empty())
    {
        if (!jacobianAtCurrentParams)
        {
            solver.reset();
            parallel_for_(Range(0, nimages),
                          JAccumulator(solver, matM, _m, npoints, param_m,
                                       flags, aspectRatio, NINTRINSIC,
                                       releaseObject, maxPoints, globalMutex));
            jacobianAtCurrentParams = true;
        }
        Mat JtJ = Mat::zeros(nparams, nparams, CV_64F);

        // Manual copy for U to avoid assertion
        Mat srcU = solver.U(Rect(0, 0, NINTRINSIC, NINTRINSIC));
        Mat dstU = JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC));
        for(int r=0; r<srcU.rows; r++) {
            const double* sptr = srcU.ptr<double>(r);
            double* dptr = dstU.ptr<double>(r);
            for(int c=0; c<srcU.cols; c++) dptr[c] = sptr[c];
        }

        for (int i = 0; i < nimages; i++)
        {
            int si = NINTRINSIC + i * 6;
            // Manual copy for V[i]
            Mat srcV = solver.V[i];
            Mat dstV = JtJ(Rect(si, si, 6, 6));
            for(int r=0; r<srcV.rows; r++) {
                const double* sptr = srcV.ptr<double>(r);
                double* dptr = dstV.ptr<double>(r);
                for(int c=0; c<srcV.cols; c++) dptr[c] = sptr[c];
            }

            Mat srcW = solver.W[i].rowRange(0, NINTRINSIC);
            Mat dstJ = JtJ(Rect(si, 0, 6, NINTRINSIC));

            // Manual copy for W part
            srcW.copyTo(dstJ);
        }

        if (releaseObject)
        {
            int obj_start = NINTRINSIC + nimages * 6;

            // Manual copy for U (extended) parts
            // Top-right block
            Mat srcU_tr = solver.U(Rect(NINTRINSIC, 0, maxPoints * 3, NINTRINSIC));
            Mat dstU_tr = JtJ(Rect(obj_start, 0, maxPoints * 3, NINTRINSIC));
            srcU_tr.copyTo(dstU_tr);

            // Bottom-right block
            Mat srcU_br = solver.U(Rect(NINTRINSIC, NINTRINSIC, maxPoints * 3, maxPoints * 3));
            Mat dstU_br = JtJ(Rect(obj_start, obj_start, maxPoints * 3, maxPoints * 3));
            srcU_br.copyTo(dstU_br);

            for (int i = 0; i < nimages; i++)
            {
                int si = NINTRINSIC + i * 6;
                // Manual copy for W extended part
                Mat srcW_ext = solver.W[i].rowRange(NINTRINSIC, solver.W[i].rows);
                Mat dstJ_ext = JtJ(Rect(obj_start, si, maxPoints * 3, 6));
                cv::transpose(srcW_ext, dstJ_ext);
            }
        }

        completeSymm(JtJ, false);

        Mat JtJinv, JtJN;
        JtJN.create(nparams_nz, nparams_nz, CV_64F);
        std::vector<uchar> mask_copy(mask.ptr<uchar>(), mask.ptr<uchar>() + nparams);
        subMatrixWithIndices(JtJ, JtJN, mask_copy, mask_copy);
        completeSymm(JtJN, false);
        cv::invert(JtJN, JtJinv, DECOMP_SVD);

        int nErrors = 2 * total - nparams_nz;
        double sigma2 = norm(allErrorsBuf, NORM_L2SQR) / nErrors;

        int j = 0;
        for (int s = 0; s < nparams; s++)
        {
            stdDevs.at<double>(s) = mask.at<uchar>(s) ?
                std::sqrt(JtJinv.at<double>(j, j) * sigma2) : 0.0;
            if (mask.at<uchar>(s))
            {
                j++;
            }
        }
    }

    // Store optimization results
    A = Matx33d(param_m(0), 0, param_m(2), 0, param_m(1), param_m(3), 0, 0, 1);
    Mat(A).convertTo(cameraMatrix, cameraMatrix.type());

    _k = Mat(distCoeffs.size(), CV_64F, param_m.ptr<double>() + 4);
    _k.convertTo(distCoeffs, distCoeffs.type());


    if (!newObjPoints.empty() && releaseObject)
    {
        int s = NINTRINSIC + nimages * 6;
        Mat _Mi = param_m.rowRange(s, s + maxPoints * 3);
        _Mi.reshape(3, 1).convertTo(newObjPoints, newObjPoints.type());

    }
    for (int i = 0; i < nimages; i++)
    {
        if (!rvecs.empty())
        {
            Mat src = param_m.rowRange(NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);
            if (rvecs.rows == nimages && rvecs.cols * rvecs.channels() == 9)
            {
                Mat dst(3, 3, rvecs.depth(), rvecs.ptr(i));
                Rodrigues(src, A);
                Mat(A).convertTo(dst, dst.type());
            }
            else
            {
                Mat dst(3, 1, rvecs.depth(), rvecs.rows == 1 ?
                    rvecs.data + i * rvecs.elemSize() : rvecs.ptr(i));
                src.convertTo(dst, dst.type());
            }
        }
        if (!tvecs.empty())
        {
            Mat src = param_m.rowRange(NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);
            Mat dst(3, 1, tvecs.depth(), tvecs.rows == 1 ?
                    tvecs.data + i * tvecs.elemSize() : tvecs.ptr(i));
            src.convertTo(dst, dst.type());
        }
    }
    return std::sqrt(reprojErr/total);
}
```


### 5.6 引擎 B（回退）：Bouguet 引擎

设 `CALIB_DISABLE_SCHUR_COMPLEMENT` 时走这条。它对标定板点做**稠密**处理——参数数少时更稳，参数多时内存和耗时都爆炸。


`subMatrixWithIndices` 按 mask 抽取雅可比的活跃行列；`LevMarq` 是经典 LM 实现，`updateAlt` 对外提供「给 `JtJ`/`JtErr` 累加」的接口，让调用方可以在循环里自己攒正规方程（这也是它比 `update` 更常用于标定的原因）。

```cpp
static void subMatrixWithIndices(const Mat& src, Mat& dst,
                                 const std::vector<uchar>& cols,
                                 const std::vector<uchar>& rows)
{
    CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);
    int m = (int)rows.size(), n = (int)cols.size();
    int i1 = 0, j1 = 0;
    for(int i = 0; i < m; i++)
    {
        if(rows[i])
        {
            const double* srcptr = src.ptr<double>(i);
            double* dstptr = dst.ptr<double>(i1++);

            if(n > 0)
            {
                const uchar* colsdata = cols.data();
                for(int j = j1 = 0; j < n; j++)
                {
                    if(colsdata[j])
                        dstptr[j1++] = srcptr[j];
                }
            }
        }
    }
}
```


`LevMarq` 的四个方法（构造函数 + `init` / `update` / `updateAlt` / `step`）：

```cpp
LevMarq::LevMarq()
{
    lambdaLg10 = 0; state = DONE;
    iters = 0;
    completeSymmFlag = false;
    errNorm = prevErrNorm = DBL_MAX;
    solveMethod = DECOMP_SVD;
}

LevMarq::LevMarq( int nparams, int nerrs, TermCriteria criteria0, bool _completeSymmFlag )
{
    init(nparams, nerrs, criteria0, _completeSymmFlag);
}

void LevMarq::init( int nparams, int nerrs, TermCriteria criteria0, bool _completeSymmFlag )
{
    mask.create( nparams, 1);
    mask.setTo(1);
    prevParam.create( nparams, 1);
    param.create(nparams, 1);
    JtJ.create(nparams, nparams);
    JtErr.create(nparams, 1);
    if( nerrs > 0 )
    {
        J.create(nerrs, nparams);
        err.create( nerrs, 1);
    }
    errNorm = prevErrNorm = DBL_MAX;
    lambdaLg10 = -3;
    criteria = criteria0;
    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = MIN(MAX(criteria.maxCount,1),1000);
    else
        criteria.maxCount = 30;
    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = MAX(criteria.epsilon, 0);
    else
        criteria.epsilon = DBL_EPSILON;
    state = STARTED;
    iters = 0;
    completeSymmFlag = _completeSymmFlag;
    solveMethod = DECOMP_SVD;
}

bool LevMarq::update( Mat1d& _param, Mat1d& matJ, Mat1d& _err )
{
    CV_Assert( !err.empty() );
    if( state == DONE )
    {
        _param = param;
        return false;
    }

    if( state == STARTED )
    {
        _param = param;
        J.setTo(0);
        err.setTo(0);
        matJ = J;
        _err = err;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        Mat(J.t() * J).copyTo(JtJ);
        JtErr = J.t() * err;
        param.copyTo(prevParam);
        step();
        if( iters == 0 )
            prevErrNorm = norm(err, NORM_L2);
        _param = param;
        err.setTo(0);
        _err = err;
        state = CHECK_ERR;
        return true;
    }

    CV_Assert( state == CHECK_ERR );
    errNorm = norm( err, NORM_L2 );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {
            step();
            _param = param;
            err.setTo(0);
            _err = err;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.maxCount ||
        norm(param, prevParam, NORM_L2 | NORM_RELATIVE) < criteria.epsilon )
    {
        _param = param;
        state = DONE;
        return true;
    }

    prevErrNorm = errNorm;
    _param = param;
    J.setTo(0);
    matJ = J;
    _err = err;
    state = CALC_J;
    return true;
}


bool LevMarq::updateAlt( Mat1d& _param, Mat1d& _JtJ, Mat1d& _JtErr, double*& _errNorm )
{
    CV_Assert( err.empty() );
    if( state == DONE )
    {
        _param = param;
        return false;
    }

    if( state == STARTED )
    {
        _param = param;
        JtJ.setTo(0);
        JtErr.setTo(0);
        errNorm = 0;
        _JtJ = JtJ;
        _JtErr = JtErr;
        _errNorm = &errNorm;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        param.copyTo(prevParam);
        step();
        _param = param;
        prevErrNorm = errNorm;
        errNorm = 0;
        _errNorm = &errNorm;
        state = CHECK_ERR;
        return true;
    }

    CV_Assert( state == CHECK_ERR );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {
            step();
            _param = param;
            errNorm = 0;
            _errNorm = &errNorm;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.maxCount ||
        norm(param, prevParam, NORM_L2 | NORM_RELATIVE) < criteria.epsilon )
    {
        _param = param;
        _JtJ = JtJ;
        _JtErr = JtErr;
        state = DONE;
        return false;
    }

    prevErrNorm = errNorm;
    JtJ.setTo(0);
    JtErr.setTo(0);
    _param = param;
    _JtJ = JtJ;
    _JtErr = JtErr;
    state = CALC_J;
    return true;
}

void LevMarq::step()
{
    using namespace cv;
    const double LOG10 = log(10.);
    double lambda = exp(lambdaLg10*LOG10);
    int nparams = param.rows;

    int nparams_nz = countNonZero(mask);
    if(JtJN.rows != nparams_nz) {
        // prevent re-allocation in every step
        JtJN.create(nparams_nz, nparams_nz);
        JtJV.create(nparams_nz, 1);
        JtJW.create(nparams_nz, 1);
    }

    subMatrixWithIndices(JtErr, JtJV, std::vector<uchar>(1, 1), mask);
    subMatrixWithIndices(JtJ, JtJN, mask, mask);

    if( err.empty() )
        completeSymm( JtJN, completeSymmFlag );

    JtJN.diag() *= 1. + lambda;
    solve(JtJN, JtJV, JtJW, solveMethod);

    int j = 0;
    for( int i = 0; i < nparams; i++ )
        param(i, 0) = prevParam(i, 0) - (mask(i, 0) ? JtJW(j++, 0) : 0);
}
```


引擎主体。注意第 2 步的 `findExtrinsicCameraParams2` —— 它就是 `solvePnP` 的内部实现：

```cpp
static double calibrateCameraInternalBouguet( const Mat& objectPoints,
                                             const Mat& imagePoints, const Mat& npoints,
                                             Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs,
                                             Mat rvecs, Mat tvecs, Mat newObjPoints, Mat stdDevs,
                                             Mat perViewErr, int flags, const TermCriteria& termCrit )
{
    int NINTRINSIC = CALIB_NINTRINSIC;
    double reprojErr = 0;

    double aspectRatio = 0.;
    int nimages = npoints.checkVector(1, CV_32S);
    CV_Assert(nimages >= 1);
    int ndistCoeffs = (int)distCoeffs.total();
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints.at<int>(0) - 1;

    // 0. check the parameters & allocate buffers
    if( imageSize.width <= 0 || imageSize.height <= 0 )
        CV_Error( cv::Error::StsOutOfRange, "image width and height must be positive" );

    if(flags & CALIB_TILTED_MODEL)
    {
        //when the tilted sensor model is used the distortion coefficients matrix must have 14 parameters
        if (ndistCoeffs != 14)
            CV_Error( cv::Error::StsBadArg, "The tilted sensor model must have 14 parameters in the distortion matrix" );
    }
    else
    {
        //when the thin prism model is used the distortion coefficients matrix must have 12 parameters
        if(flags & CALIB_THIN_PRISM_MODEL)
            if (ndistCoeffs != 12)
                CV_Error( cv::Error::StsBadArg, "Thin prism model must have 12 parameters in the distortion matrix" );
    }

    if( !rvecs.empty() )
    {
        int cn = rvecs.channels();
        CV_Assert(rvecs.depth() == CV_32F || rvecs.depth() == CV_64F);
        CV_Assert((rvecs.rows == nimages && (rvecs.cols*cn == 3 || rvecs.cols*cn == 3)) ||
                  (rvecs.rows == 1 && rvecs.cols == nimages && cn == 3));
    }

    if( !tvecs.empty() )
    {
        int cn = tvecs.channels();
        CV_Assert(tvecs.depth() == CV_32F || tvecs.depth() == CV_64F);
        CV_Assert((tvecs.rows == nimages && tvecs.cols*cn == 3) ||
                  (tvecs.rows == 1 && tvecs.cols == nimages && cn == 3));
    }

    CV_Assert(cameraMatrix.type() == CV_32F || cameraMatrix.type() == CV_64F);
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);

    CV_Assert(distCoeffs.type() == CV_32F || distCoeffs.type() == CV_64F);
    CV_Assert(distCoeffs.rows == 1 || distCoeffs.cols == 1);
    CV_Assert(ndistCoeffs == 4 || ndistCoeffs == 5 || ndistCoeffs == 8 ||
              ndistCoeffs == 12 || ndistCoeffs == 14);

    int total = 0, maxPoints = 0;
    for(int i = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);
        if( ni < 4 )
        {
            CV_Error_( cv::Error::StsOutOfRange, ("The number of points in the view #%d is < 4", i));
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    if( !newObjPoints.empty() )
    {
        int cn = newObjPoints.channels();
        CV_Assert(newObjPoints.depth() == CV_32F || newObjPoints.depth() == CV_64F);
        CV_Assert((newObjPoints.rows == maxPoints && newObjPoints.cols*cn == 3) ||
                  (newObjPoints.rows == 1 && newObjPoints.cols == maxPoints && cn == 3));
    }

    if( !stdDevs.empty() )
    {
        int cn = stdDevs.channels();
        CV_Assert(stdDevs.depth() == CV_32F || stdDevs.depth() == CV_64F);
        int nstddev = nimages*6 + NINTRINSIC + (releaseObject ? maxPoints*3 : 0);

        CV_Assert((stdDevs.rows == nstddev && stdDevs.cols*cn == 1) ||
                  (stdDevs.rows == 1 && stdDevs.cols == nstddev && cn == 1));
    }

    Mat matM( 1, total, CV_64FC3 );
    Mat _m( 1, total, CV_64FC2 );
    Mat allErrors(1, total, CV_64FC2);

    if(objectPoints.channels() == 3)
        objectPoints.convertTo(matM, CV_64F);
    else {
        convertPointsToHomogeneous(objectPoints, matM);
        matM.convertTo(matM, CV_64F);
    }

    if(imagePoints.channels() == 2)
        imagePoints.convertTo(_m, CV_64F);
    else {
        convertPointsFromHomogeneous(imagePoints, _m);
        _m.convertTo(_m, CV_64F);
    }

    int nparams = NINTRINSIC + nimages*6;
    if( releaseObject )
        nparams += maxPoints * 3;

    std::vector<double> k(14, 0.0);
    Mat _k( distCoeffs.rows, distCoeffs.cols, CV_64F, k.data());
    if( distCoeffs.total() < 8 )
    {
        if( distCoeffs.total() < 5 )
            flags |= CALIB_FIX_K3;
        flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
    }
    const double minValidAspectRatio = 0.01;
    const double maxValidAspectRatio = 100.0;

    Matx33d A;
    cameraMatrix.convertTo(A, CV_64F);

    // 1. initialize intrinsic parameters & LM solver
    if( flags & CALIB_USE_INTRINSIC_GUESS )
    {
        if( A(0, 0) <= 0 || A(1, 1) <= 0 )
            CV_Error( cv::Error::StsOutOfRange, "Focal length (fx and fy) must be positive" );
        if( A(0, 2) < 0 || A(0, 2) >= imageSize.width ||
            A(1, 2) < 0 || A(1, 2) >= imageSize.height )
            CV_Error( cv::Error::StsOutOfRange, "Principal point must be within the image" );
        if( fabs(A(0, 1)) > 1e-5 )
            CV_Error( cv::Error::StsOutOfRange, "Non-zero skew is not supported by the function" );
        if( fabs(A(1, 0)) > 1e-5 || fabs(A(2, 0)) > 1e-5 ||
            fabs(A(2, 1)) > 1e-5 || fabs(A(2,2)-1) > 1e-5 )
            CV_Error( cv::Error::StsOutOfRange,
                "The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape" );
        A(0, 1) = A(1, 0) = A(2, 0) = A(2, 1) = 0.;
        A(2, 2) = 1.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = A(0, 0)/A(1, 1);

            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        distCoeffs.convertTo(_k, CV_64F);
    }
    else
    {
        Scalar mean, sdv;
        meanStdDev(matM, mean, sdv);
        if( fabs(mean[2]) > 1e-5 || fabs(sdv[2]) > 1e-5 )
            CV_Error( cv::Error::StsBadArg,
            "For non-planar calibration rigs the initial intrinsic matrix must be specified" );
        for(int i = 0; i < total; i++ )
            matM.at<Point3d>(i).z = 0.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = A(0, 0);
            aspectRatio /= A(1, 1);
            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        initIntrinsicParams2D( matM, _m, npoints, imageSize, A, aspectRatio );
    }

    LevMarq solver( nparams, 0, termCrit );

    if(flags & CALIB_USE_LU) {
        solver.solveMethod = DECOMP_LU;
    }
    else if(flags & CALIB_USE_QR) {
        solver.solveMethod = DECOMP_QR;
    }

    {
    double* param = solver.param.ptr<double>();
    uchar* mask = solver.mask.ptr<uchar>();

    param[0] = A(0, 0); param[1] = A(1, 1); param[2] = A(0, 2); param[3] = A(1, 2);
    std::copy(k.begin(), k.end(), param + 4);

    if(flags & CALIB_FIX_ASPECT_RATIO)
        mask[0] = 0;
    if( flags & CALIB_FIX_FOCAL_LENGTH )
        mask[0] = mask[1] = 0;
    if( flags & CALIB_FIX_PRINCIPAL_POINT )
        mask[2] = mask[3] = 0;
    if( flags & CALIB_ZERO_TANGENT_DIST )
    {
        param[6] = param[7] = 0;
        mask[6] = mask[7] = 0;
    }
    if( !(flags & CALIB_RATIONAL_MODEL) )
        flags |= CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6;
    if( !(flags & CALIB_THIN_PRISM_MODEL))
        flags |= CALIB_FIX_S1_S2_S3_S4;
    if( !(flags & CALIB_TILTED_MODEL))
        flags |= CALIB_FIX_TAUX_TAUY;

    mask[ 4] = !(flags & CALIB_FIX_K1);
    mask[ 5] = !(flags & CALIB_FIX_K2);
    if( flags & CALIB_FIX_TANGENT_DIST )
        mask[6]  = mask[7]  = 0;
    mask[ 8] = !(flags & CALIB_FIX_K3);
    mask[ 9] = !(flags & CALIB_FIX_K4);
    mask[10] = !(flags & CALIB_FIX_K5);
    mask[11] = !(flags & CALIB_FIX_K6);

    if(flags & CALIB_FIX_S1_S2_S3_S4)
    {
        mask[12] = 0;
        mask[13] = 0;
        mask[14] = 0;
        mask[15] = 0;
    }
    if(flags & CALIB_FIX_TAUX_TAUY)
        mask[16] = mask[17] = 0;

    if(releaseObject)
    {
        // copy object points
        int s = NINTRINSIC + nimages * 6;

        std::copy( matM.ptr<double>(), matM.ptr<double>( 0, maxPoints - 1 ) + 3,
                   param + NINTRINSIC + nimages * 6 );
        // fix points
        mask[s + 0] = 0;
        mask[s + 1] = 0;
        mask[s + 2] = 0;
        mask[s + iFixedPoint * 3 + 0] = 0;
        mask[s + iFixedPoint * 3 + 1] = 0;
        mask[s + iFixedPoint * 3 + 2] = 0;
        mask[nparams - 1] = 0;
    }
    }

    Mat_<double> param_m = solver.param;
    Mat mask = solver.mask;
    int nparams_nz = countNonZero(mask);

    if (nparams_nz >= 2 * total)
        CV_Error_(Error::StsBadArg,
                  ("There should be less vars to optimize (having %d) than the number of residuals (%d = 2 per point)", nparams_nz, 2 * total));

    // 2. initialize extrinsic parameters
    for(int i = 0, pos = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);

        int s = NINTRINSIC + i*6;
        Mat _ri = param_m.rowRange(s, s + 3);
        Mat _ti = param_m.rowRange(s + 3, s + 6);

        Mat _Mi = matM.colRange(pos, pos + ni);
        Mat _mi = _m.colRange(pos, pos + ni);

        findExtrinsicCameraParams2( _Mi, _mi, Mat(A), _k, _ri, _ti, /*useExtrinsicGuess=*/0 );

        pos += ni;
    }

    // 3. run the optimization

    Mat errBuf( maxPoints*2, 1, CV_64FC1 );
    Mat JiBuf ( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
    Mat JeBuf ( maxPoints*2, 6, CV_64FC1 );
    Mat JoBuf;
    if (releaseObject)
        JoBuf = Mat( maxPoints*2, maxPoints*3, CV_64FC1);

    for(;;)
    {
        bool optimizeObjPoints = releaseObject;

        Mat1d _param;
        Mat1d JtJ, JtErr;
        double* _errNorm = 0;
        bool proceed = solver.updateAlt( _param, JtJ, JtErr, _errNorm );
        double *param = _param.ptr<double>(), *pparam = solver.prevParam.ptr<double>();
        bool calcJ = solver.state == LevMarq::CALC_J || (!proceed && !stdDevs.empty());

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            param[0] = param[1]*aspectRatio;
            pparam[0] = pparam[1]*aspectRatio;
        }

        double fx = param_m(0), fy = param_m(1), cx = param_m(2), cy = param_m(3);
        Matx33d intrin(fx,  0, cx,
                        0, fy, cy,
                        0,  0,  1);
        Mat dist = param_m.rowRange(4, 4+14);

        if ( !proceed && stdDevs.empty() && perViewErr.empty() )
            break;
        else if ( !proceed && !stdDevs.empty() )
            JtJ.setTo(0);

        reprojErr = 0;

        int so = NINTRINSIC + nimages * 6;
        int pos = 0;
        for( int i = 0; i < nimages; i++ )
        {
            int si = NINTRINSIC + i * 6;

            int ni = npoints.at<int>(i);
            Mat _ri = param_m.rowRange(si, si + 3);
            Mat _ti = param_m.rowRange(si + 3, si + 6);

            Mat _Mi = matM.colRange(pos, pos + ni);
            if( optimizeObjPoints )
            {
                _Mi = param_m.rowRange(so, so + ni * 3);
                _Mi = _Mi.reshape(3, 1);
            }
            Mat _mi = _m.colRange(pos, pos + ni);
            Mat _me = allErrors.colRange(pos, pos + ni);

            Mat Jo;
            if (optimizeObjPoints)
                Jo = JoBuf(Range(0, ni*2), Range(0, ni*3));

            Mat Je  = JeBuf.rowRange(0, ni*2);
            Mat Ji  = JiBuf.rowRange(0, ni*2);
            Mat err = errBuf.rowRange(0, ni*2);
            Mat _mp = err.reshape(2, 1);

            if( calcJ )
            {
                Mat _dpdr = Je.colRange(0, 3);
                Mat _dpdt = Je.colRange(3, 6);
                Mat _dpdf = (flags & CALIB_FIX_FOCAL_LENGTH) ? Mat() : Ji.colRange(0, 2);
                Mat _dpdc = (flags & CALIB_FIX_PRINCIPAL_POINT) ? Mat() : Ji.colRange(2, 4);
                Mat _dpdk = Ji.colRange(4, NINTRINSIC);
                Mat _dpdo = Jo.empty() ? Mat() : Jo.colRange(0, ni * 3);

                projectPoints(_Mi, _ri, _ti, intrin, dist, _mp, _dpdr, _dpdt,
                              (_dpdf.empty() ? noArray() : _dpdf),
                              (_dpdc.empty() ? noArray() : _dpdc),
                              _dpdk, (_dpdo.empty() ? noArray() : _dpdo),
                              (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0.);
            }
            else
                projectPoints( _Mi, _ri, _ti, intrin, dist, _mp);

            subtract( _mp, _mi, _mp );
            _mp.copyTo(_me);

            if( calcJ )
            {
                // see HZ: (A6.14) for details on the structure of the Jacobian
                JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += Ji.t() * Ji;
                JtJ(Rect(si, si, 6, 6)) = Je.t() * Je;
                JtJ(Rect(si,  0, 6, NINTRINSIC)) = Ji.t() * Je;
                if( optimizeObjPoints )
                {
                    JtJ(Rect(so, 0, maxPoints * 3, NINTRINSIC)) += Ji.t() * Jo;
                    JtJ(Rect(so, si, maxPoints * 3, 6)) += Je.t() * Jo;
                    JtJ(Rect(so, so, maxPoints * 3, maxPoints * 3)) += Jo.t() * Jo;
                }

                JtErr.rowRange(0, NINTRINSIC) += Ji.t() * err;
                JtErr.rowRange(si, si + 6) = Je.t() * err;
                if( optimizeObjPoints )
                {
                    JtErr.rowRange(so, nparams) += Jo.t() * err;
                }
            }

            double viewErr = norm(err, NORM_L2SQR);
            if( !perViewErr.empty() )
                perViewErr.at<double>(i) = std::sqrt(viewErr / ni);

            reprojErr += viewErr;
            pos += ni;
        }
        if( _errNorm )
            *_errNorm = reprojErr;

        if( !proceed )
        {
            if( !stdDevs.empty() )
            {
                Mat JtJinv, JtJN;
                JtJN.create(nparams_nz, nparams_nz, CV_64F);
                subMatrixWithIndices(JtJ, JtJN, mask, mask);
                completeSymm(JtJN, false);
                cv::invert(JtJN, JtJinv, DECOMP_SVD);
                // an explanation of that denominator correction can be found here:
                // R. Hartley, A. Zisserman, Multiple View Geometry in Computer Vision, 2004, section 5.1.3, page 134
                // see the discussion for more details: https://github.com/opencv/opencv/pull/22992
                int nErrors = 2 * total - nparams_nz;
                double sigma2 = norm(allErrors, NORM_L2SQR) / nErrors;
                int j = 0;
                for ( int s = 0; s < nparams; s++ )
                {
                    stdDevs.at<double>(s) = mask.data[s] ? std::sqrt(JtJinv.at<double>(j,j) * sigma2) : 0.0;
                    if( mask.data[s] )
                        j++;
                }
            }
            break;
        }
    }

    // 4. store the results
    double * param = solver.param.ptr<double>();
    A = Matx33d(param[0], 0, param[2], 0, param[1], param[3], 0, 0, 1);
    Mat(A).convertTo(cameraMatrix, cameraMatrix.type());
    _k = Mat(distCoeffs.size(), CV_64F, param + 4);
    _k.convertTo(distCoeffs, distCoeffs.type());

    if( !newObjPoints.empty() && releaseObject )
    {
        int s = NINTRINSIC + nimages * 6;
        Mat _Mi = param_m.rowRange(s, s + maxPoints * 3);
        _Mi.reshape(3, 1).convertTo(newObjPoints, newObjPoints.type());
    }

    for(int i = 0; i < nimages; i++ )
    {
        if( !rvecs.empty() )
        {
            Mat src = Mat(3, 1, CV_64F, param + NINTRINSIC + i*6);
            if( rvecs.rows == nimages && rvecs.cols*rvecs.channels() == 9 )
            {
                Mat dst(3, 3, rvecs.depth(), rvecs.ptr(i));
                Rodrigues(src, A);
                Mat(A).convertTo(dst, dst.type());
            }
            else
            {
                Mat dst(3, 1, rvecs.depth(), rvecs.rows == 1 ?
                    rvecs.data + i*rvecs.elemSize() : rvecs.ptr(i));
                src.convertTo(dst, dst.type());
            }
        }
        if( !tvecs.empty() )
        {
            Mat src(3, 1, CV_64F, param + NINTRINSIC + i*6 + 3);
            Mat dst(3, 1, tvecs.depth(), tvecs.rows == 1 ?
                    tvecs.data + i*tvecs.elemSize() : tvecs.ptr(i));
            src.convertTo(dst, dst.type());
        }
    }

    return std::sqrt(reprojErr/total);
}
```


---

## 6. 常用 flag 速查

| Flag | 值 | 作用 |
| --- | --- | --- |
| `CALIB_USE_INTRINSIC_GUESS` | `0x00001` | 用传入的 `cameraMatrix` 做初值，而非内部估计 |
| `CALIB_FIX_PRINCIPAL_POINT` | `0x00004` | 固定主点 |
| `CALIB_FIX_ASPECT_RATIO` | `0x00002` | 固定 `fx/fy` 比例（内参初值里的 `aspectRatio`） |
| `CALIB_ZERO_TANGENT_DIST` | `0x00008` | 切向畸变 `p1=p2=0` 且不再优化 |
| `CALIB_FIX_FOCAL_LENGTH` | `0x00010` | 固定焦距 |
| `CALIB_RATIONAL_MODEL` | `0x04000` | 启用 `k4..k6`（8 维畸变） |
| `CALIB_THIN_PRISM_MODEL` | `0x08000` | 启用 `s1..s4`（12 维畸变） |
| `CALIB_TILTED_MODEL` | `0x40000` | 启用 `tauX, tauY`（14 维畸变） |
| `CALIB_USE_QR` | `0x100000` | 用 QR 代替 SVD 解线性方程，更快但精度略低 |
| `CALIB_USE_LU` | `1 << 17` | 用 LU，更快但精度更低 |
| `CALIB_DISABLE_SCHUR_COMPLEMENT` | `1 << 23` | 关掉 Schur 补，回退到 Bouguet 引擎 |

> 注意源码里这段默认值逻辑：不传 `CALIB_RATIONAL_MODEL` 就会自动加上 `CALIB_FIX_K4|K5|K6`，不传 `CALIB_THIN_PRISM_MODEL` 就自动加 `CALIB_FIX_S1_S2_S3_S4`，不传 `CALIB_TILTED_MODEL` 就自动加 `CALIB_FIX_TAUX_TAUY`。换句话说「关掉某个模型」不是靠减参数，而是靠把对应 mask 位清零。


---

## 7. 参考资料

- 源码：[opencv/opencv · modules/calib3d/src/calibration.cpp](https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration.cpp)
- 公开声明与文档：[modules/calib3d/include/opencv2/calib3d.hpp](https://github.com/opencv/opencv/blob/master/modules/calib3d/include/opencv2/calib3d.hpp)（`calibrateCamera` 在 1743 行起）
- 官方教程：[Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- 官方教程（Python 实例）：[Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Z. Zhang, A Flexible New Technique for Camera Calibration, TPAMI 2000](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)（`initIntrinsicParams2D` 的理论来源）
- Triggs et al., Bundle Adjustment — A Modern Synthesis, 1999（Schur 补 BA）
- Lourakis & Argyros, SBA: A Software Package for Generic Sparse Bundle Adjustment, 2009
- Strobl et al., ICCV 2011（release-object 方法，对应 `calibrateCameraRO`）