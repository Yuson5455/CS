#coding:utf-8
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DCT基作为稀疏基，重建算法为OMP算法 ，图像按列进行处理
# 参考文献: 任晓馨. 压缩感知贪婪匹配追踪类重建算法研究[D].
#北京交通大学, 2012.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 导入所需的第三方库文件
import  numpy as np
import math
from PIL import Image

#读取图像，并变成numpy类型的 array
im = np.array(Image.open('lena.jpg')) #图片大小256*256

#生成高斯随机测量矩阵
sampleRate=0.7  #采样率
Phi=np.random.randn(math.floor(256*sampleRate),256)

#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((256,256))
v=list(range(256))
for k in range(0,256):
    dct_1d=np.cos(np.dot(v,k*math.pi/256))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#随机测量
img_cs_1d=np.dot(Phi,im)

#OMP算法函数
def cs_omp(y,D):
    L = int(math.floor(3*(y.shape[0])//4))
    residual=y  #初始化残差
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((256))
    for j in range(L):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置
        index[j]=pos
        my=np.linalg.pinv(D[:,0:(index>=0).sum()]) #最小二乘,看参考文献1
        # print('index is --> '+str(index))
        # print('my.shape -->'+str(my.shape))


        a=np.dot(my,y) #最小二乘,看参考文献1
        # print('a.shape-->' + str(a.shape))
        residual=y-np.dot(D[:,:(index>=0).sum()],a)
        # print("residual.shape" +str(residual.shape))
    result[0:(index>=0).sum()]= a
    # print('result.shape -->' + str(result.shape))
    return  result

#重建
sparse_rec_1d=np.zeros((256,256))   # 初始化稀疏系数矩阵
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(256):
    print(('rebuilding',i,'th column'))
    column_rec=cs_omp(img_cs_1d[:,i],Theta_1d) #利用OMP算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)
image2.show()