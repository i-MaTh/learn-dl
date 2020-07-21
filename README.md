# Understanding deep learning
基于传统优化方法的3D人脸重建，优化目标为
$$\argmin_{s,R,T,\alpha_{id}, \alpha_{exp}}\sum_{k=1}K\Vert( s\cdot R\cdot (\bar{M}+A_{id}\alpha_{id}+A_{exp}\alpha_{exp}){v_k} +T)-L_k\Vert + \lambda\Vert \mathbf{p}\Vert_\Lambda \tag1$$
其中，$s$为缩放系数，$R$为旋转矩阵，$T=[T_x,T_y]^T$为平面平移量，$\alpha_{id}$为形状参数，$\alpha_{exp}$为表情参数，${\bar{M}, A_{id}, A_{exp}}$为3DMM基底，$L$为2D配准点坐标。$K$为2D人脸配准点的个数，$v_k$表示3D稠密的模型中与第$k$个2D配准点语义一致的3D顶点的索引。$\mathbf{p}=[s,R,T,\alpha_{id}, \alpha_{exp}]$为所有未知量，$\Lambda$为不同未知量的权重。
