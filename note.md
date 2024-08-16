# CV



## Lec 1

### Part3. History

#### Pre-History

+ Da Vinci: perspective projection 透视投影
+ Daguerreotype: 银版照相
+ Great Trigonometrical Survey

#### Recent

+ s
+ Rosenblatt's perception
+ Block World; from topological 
+ symbolic way. useless now
+ shape from shading
+  intrinsic images -- several 2D layers components

+ photometric stereo -- multiple 2D images

+ essential matrix --  通过两张图片3D重建
+ binocular
+ dense optical flow -- 追踪图序列的相邻帧
+ markov random fields 先验知识编码+推理视觉问题+全局优化

+ part-based models -- 简单基础模块搭建3D模型
+ backpropagation algorithm 
+ self-driving car -- prediction or neural network
+ structure from motion -- 光度测距，二维图像序列估计三维结构
+ iterative closest points -- 两组点云，固定+迭代估计匹配，对齐三维模型
+ volumetric fusion -- 收集、处理像素数据，像素集转移到体素集，计算有向距离（到最近表面），所有数据融合到同一模型，最后做表面提取重建3D模型
+ multi-view stereo

+ conv net
+ morphable model
+ SIFT feature detecting
+ photo tourism -- 通过照片做大场景重建
+ patch-based multi view stereo -- 通过区域而非像素做重建
+ kinect -- 3D 动作识别
+ ImageNet and AlexNet with GPU training
+ datasets' importance

+ visualization of learning of network
+ generative adversarial network

+ deep reinforcement learning



## Lec 2

#### 2D points

homogeneous coordinates:
$$
\begin{bmatrix}
 x \\
 y \\
 w 
\end{bmatrix}
$$
P<sup>2</sup> = R<sup>3</sup> \ {(0, 0, 0)}, and define only up to scale.

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330160651839.png" alt="image-20240330160651839" style="zoom:50%;" />

conversion between homogeneous and inhomogeneous coordinates with augmented coordinate -- the distinguished equivalent x (simplify w = 1).

homogeneous coordinate with w = 0 symbolize the infinite point.

#### 2D lines

**I** = (a, b, c)<sup>T</sup> , with normalize : **I** = (n<Sub>x</sub>, n<sub>y</sub>, d)<sup>T</sup> where |n|<sub>2</sub> = 1 && d = distance to origin

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330161035435.png" alt="image-20240330161035435" style="zoom:50%;" />

**I<sub>∞</sub>** : infinity lines as (0, 0, 1)



#### 2D Cross Product

向量叉乘可以用向量a的斜对称矩阵和向量b做矩阵乘法表示

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330162640971.png" alt="image-20240330162640971" style="zoom:50%;" />

**I** = **X1** × **X2** 两点连线

**X** = **I1** × **I2** 两线交点

表达圆锥截面<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330163434413.png" alt="image-20240330163434413" style="zoom:50%;" />



#### 2D transformation

Transformation：<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330164422648.png" alt="image-20240330164422648" style="zoom:50%;" />

Euclidian：<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330164652750.png" alt="image-20240330164652750" style="zoom:50%;" /> with R ∈ SO2 正交旋转矩阵[<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330164818594.png" alt="image-20240330164818594" style="zoom:15%;" />]

此处，沿x轴、y轴、z轴分别旋转a°、b°、c°的乘法矩阵分别为：
$$
\begin{bmatrix}
1&0&0\\
0&cosa&-sina\\
0&sina&cosa\\
\end{bmatrix}
\begin{bmatrix}
cosb&0&sinb\\
0&1&0\\
-sinb&0&cosb\\
\end{bmatrix}
\begin{bmatrix}
cosc&-sinc&0\\
sinc&cosc&0\\
0&0&1\\
\end{bmatrix}
$$
Similarity：<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330164954118.png" alt="image-20240330164954118" style="zoom:50%;" />

Affine：<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330165039061.png" alt="image-20240330165039061" style="zoom:50%;" /> with A an arbitrary 2*2 matrix

Perspective：<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330165219981.png" alt="image-20240330165219981" style="zoom:50%;" /> with H an arbitrary 3*3 homogeneous matrix

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330185000298.png" alt="image-20240330185000298" style="zoom:50%;" />

#### homography

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240331104614949.png" alt="image-20240331104614949" style="zoom:50%;" />

+ 由于xi‘与xi均为齐次坐标，故xi’与Hxi共线，共线向量叉乘为零
+ 用h<sub>k</sub><sup>T</sup>表示H的转置的第k列，即H的第k行，为行向量

$$
H = 
\begin{bmatrix}
   a_{1} & a_{2} & a_{3} \\
   a_{4} & a_{5} & a_{6} \\
   a_{7} & a_{8} & a_{9} = 1
\end{bmatrix}
,
h_{1} = 
\begin{bmatrix}
   a_{1} & a_{2} & a_{3} \\
   a_{4} & a_{5} & a_{6} \\
   a_{7} & a_{8} & a_{9} = 1
\end{bmatrix}
$$


$$
\widetilde{x_{i}}' × H\widetilde{x_{i}} = 
\begin{bmatrix}
   0 & z' & -y' \\
   -z' & 0 & x' \\
   y' & -x' & 0
\end{bmatrix}
\begin{bmatrix}
   h_{1}^{T}\widetilde{x_{i}} \\
   h_{2}^{T}\widetilde{x_{i}} \\
   h_{3}^{T}\widetilde{x_{i}}
\end{bmatrix} =
\begin{bmatrix}
   z'h_{2}^{T}\widetilde{x_{i}} - y'h_{3}^{T}\widetilde{x_{i}} \\
   x'h_{3}^{T}\widetilde{x_{i}} - z'h_{1}^{T}\widetilde{x_{i}} \\
   y'h_{1}^{T}\widetilde{x_{i}} - x'h_{2}^{T}\widetilde{x_{i}}
\end{bmatrix}
$$

+ 将结果中h<sub>k</sub><sup>T</sup>取出，排列为1*9的向量h

$$
\widetilde{x_{i}}' × H\widetilde{x_{i}} =
\begin{bmatrix}
   0^{T} & z'\widetilde{x_{i}} & -y'\widetilde{x_{i}} \\
   -z'\widetilde{x_{i}} & 0 & x'\widetilde{x_{i}} \\
   y'\widetilde{x_{i}} & -x'\widetilde{x_{i}} & 0
\end{bmatrix}
\begin{bmatrix}
   h_{1} \\
   h_{2} \\
   h_{3}
\end{bmatrix}
$$

+ z = z' = 1, 且由于线性相关可化简掉第三行，展开得到：

$$
\widetilde{x_{i}}' × H\widetilde{x_{i}} =
\begin{bmatrix}
   0 & 0 & 0 & x_{i} & y_{i} & 1 & -y'x_{i} & -y'y_{i} & -y'\\
   -x_{i} & -y_{i} & -1 & 0 & 0 & 0 & x'x_{i} & x'y_{i} & x'
\end{bmatrix}
\begin{bmatrix}
   h_{1} \\
   h_{2} \\
   h_{3}
\end{bmatrix}
$$

+ 求解

  + 由于有八个未知量，起码需要四组点求解：

  <img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240331104835934.png" alt="image-20240331104835934" style="zoom:50%;" />

  + 但若有N组点，则用SVD解决，V最后一行即解

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240331111927183.png" alt="image-20240331111927183" style="zoom:50%;" />



#### 3D points

same as 2D， just analogy

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330163926032.png" alt="image-20240330163926032" style="zoom:50%;" />

#### 3D lines

两面交线

圆锥曲面<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240330164108372.png" alt="image-20240330164108372" style="zoom:50%;" />



### Geo Image Formation

physical/mathematical camera modal

orthographic(precisely 1:1)正投影/perspective projection透视投影

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240331112808389.png" alt="image-20240331112808389" style="zoom:30%;" />

离物越远，焦距越长，越趋于正投影

**orthographic projection:**

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417201554243.png" alt="image-20240417201554243" style="zoom:50%;" /><img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417201629223.png" alt="image-20240417201629223" style="zoom:40%;" />

正投影，压缩z轴信息为无用，x、y直接投影



**perspective projection：**

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417201932195.png" alt="image-20240417201932195" style="zoom:50%;" /><img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417201747884.png" alt="image-20240417201747884" style="zoom:40%;" />

透视投影将相机坐标系中物体投影到焦距平面的二维坐标系上。

however，上述模型建立在相机坐标系的坐标原点和二维坐标系的坐标原点均在主光轴上。若非如此，则需要有一个偏移量cx，cy。再考虑到x轴y轴的微小正交偏差s，将上述方程弥补完整：

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417203809287.png" alt="image-20240417203809287" style="zoom:50%;" />其中左侧3×3矩阵为calibration matrix **K** 标定矩阵, 整个矩阵为projection matrix.

**3\*4 projection matrix**

然而，上述只是intrinsic。考虑到相机坐标系和现实坐标系不一定相同，其中亦存在一个变化。该基向量之间的转化用extrinsic矩阵[R, t]来表示。则有如下的从现实坐标系到二维坐标系的转化流程：

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417204058971.png" alt="image-20240417204058971" style="zoom:50%;" />

将其4*4完全展开为如下形式：

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417204327860.png" alt="image-20240417204327860" style="zoom:50%;" />

在真个过程中，我们忽略了z轴信息，即inverse-depth。如若知道了z，则可根据如下关系来从二维坐标逆向回推至现实世界坐标<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240417204513591.png" alt="image-20240417204513591" style="zoom:50%;" />

然而，图像会有distortion。处理办法如下：

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240331200640958.png" alt="image-20240331200640958" style="zoom:50%;" />

projection with distortion 





### L Image Formation

camera use lense to accumulate light instead of a hole to avoid blur  

aperture 光圈； circle of confusion 弥散圆； DOF 景深； 光圈越大，景深越小

chromatic aberration 不同光波长不同，导致成像位置有细微z向偏差。

vignetting 渐晕 边缘亮度降低 -- can be calibrated 



### Image Sensing Pipeline

RGB color filter, one color on each pixel. sensitive to green -- more green.

gamma compression -- people are more sensitive to color differences in darker area.



## Lec 3	Structure-from-Motion

### Preliminary

**calibration(相机标定): **

+ capture from different poses
+ capture the features
+ optimized with both intrinsic and extrinsic
  + closed-form
  + non-linear solution

+ 多角度拍摄图像中的特征点是2D点image point；现实中基于物体平面建模，对应的特征点在现实坐标系下(简单情况，如棋盘，以平面建模即可忽略z维，设其为0)的坐标是3D点object point。通过多组对应点的2D，3D坐标对应，可以用最小二乘解出内外部参数。而这一过程中需要克服可能的畸变(径向与切向)。于此有不同方法；如opencv的五参数标定。

**Point Feature**

+ feature

  + invariant to perspective effects and illumination -- 不受透视和亮度影响

  + 不同视角下向量相似

+ detect -- SIFT

  + Gaussian -- blur -- can be scaled
  + in same scale, find differences of Gauss (DoG) of the adjacent
  + interest point -- blob -- extrema -- 层内比较，层间比较
  + for each blob, rotate to align with dominant gradient
  + construct gradient histogram
  + concatenated and normalized to a 128D vector
  + 特征匹配：最近邻搜索 -- A中特征向量以最近欧氏距离为标准匹配B中特征向量；Ratio-Test -- 用最近邻比第二近邻，比值<D.Lowe推荐值0.8则保留




### 2form structure

<img src="C:\Users\syp\AppData\Roaming\Typora\typora-user-images\image-20240403200702364.png" alt="image-20240403200702364" style="zoom:50%;" />

+ 知道两个相机视角，知道R|t，对于3D点，已知其在x1面上的位置，其在x2面的对应位置只需在epipolar line上找。





