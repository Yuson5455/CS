#coding:utf-8
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DCT基作为稀疏基，重建算法为CoSaMP算法,图像按列进行处理
# 参考文献: D. Deedell andJ. Tropp, “COSAMP: Iterative Signal Recovery from
#Incomplete and Inaccurate Samples,” 2008.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#导入集成库
import math

# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包


#读取图像，并变成numpy类型的 array
im = np.array(Image.open('lena.jpg'))#图片大小256*256

#生成高斯随机测量矩阵
sampleRate=0.7  #采样率
Phi=np.random.randn(math.floor(256*sampleRate),256)
# Phi=np.random.randn(256,256)
u, s, vh = np.linalg.svd(Phi)
# Phi = u[:256*sampleRate,] #将测量矩阵正交化


#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((256,256))
v=range(256)
for k in range(0,256):
    dct_1d=np.cos(np.dot(v,k*math.pi/256))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#随机测量
img_cs_1d=np.dot(Phi,im)

#CoSaMP算法函数
def cs_CoSaMP(y,D):
    S=math.floor(y.shape[0]/4)  #稀疏度
    residual=y  #初始化残差
    pos_last=np.array([],dtype=np.int64)
    result=np.zeros((256))

    for j in range(S):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos_temp=np.argsort(product)
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos_temp=pos_temp[0:2*S]#对应步骤3
        pos=np.union1d(pos_temp,pos_last)

        result_temp=np.zeros((256))
        result_temp[pos]=np.dot(np.linalg.pinv(D[:,pos]),y)

        pos_temp=np.argsort(np.fabs(result_temp))
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        result[pos_temp[:S]]=result_temp[pos_temp[:S]]
        pos_last=pos_temp
        residual=y-np.dot(D,result)

    return  result



#重建
sparse_rec_1d=np.zeros((256,256))   # 初始化稀疏系数矩阵
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(256):
    print('正在重建第',i,'列。。。')
    column_rec=cs_CoSaMP(img_cs_1d[:,i],Theta_1d)  #利用CoSaMP算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)
image2.show()