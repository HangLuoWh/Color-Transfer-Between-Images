import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io,color

def CT(src,trt):
    '''
    src:source image
    trt:target image
    '''
    src=src/255
    trt=trt/255
    trt_shape=trt.shape

    #reshape images and put each channel in a row
    #note that 'src.reshape(-1,3)' cannot seperate each color channel automatically
    #so I seperated color channels first, and then reshaped them to one row.
    r_s=src[:,:,0].reshape(1,-1)
    g_s=src[:,:,1].reshape(1,-1)
    b_s=src[:,:,2].reshape(1,-1)
    src_re=np.concatenate([r_s,g_s,b_s],axis=0)

    r_t=trt[:,:,0].reshape(1,-1)
    g_t=trt[:,:,1].reshape(1,-1)
    b_t=trt[:,:,2].reshape(1,-1)
    trt_re=np.concatenate([r_t,g_t,b_t],axis=0)

    #tranfer to logarithmic LMS
    trans_LMS=np.array([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]])
    src_lLMS=np.log10(np.dot(trans_LMS,src_re)+np.finfo(np.float64).eps)#to avoid log(0),we add an eps
    trt_lLMS=np.log10(np.dot(trans_LMS,trt_re)+np.finfo(np.float64).eps)#to avoid log(0),we add an eps
    #transfer to l_alpha_beta
    trans_l_al_be=np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                            [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
                            [1/np.sqrt(2), -1/np.sqrt(2), 0]])
    src_l_al_be=np.dot(trans_l_al_be, src_lLMS)
    trt_l_al_be=np.dot(trans_l_al_be, trt_lLMS)

    # statistic transfer
    src_mean=np.mean(src_l_al_be, axis=1).reshape(3,1)
    src_std=np.std(src_l_al_be, axis=1).reshape(3,1)
    trt_mean=np.mean(trt_l_al_be, axis=1).reshape(3,1)
    trt_std=np.std(trt_l_al_be, axis=1).reshape(3,1)
    result_l_al_be=(trt_l_al_be-trt_mean)/trt_std*src_std+src_mean
    
    #reverse conversion
    trans_l_al_be2lLMS=np.array([[1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)],
                                [1/np.sqrt(3), 1/np.sqrt(6), -1/np.sqrt(2)],
                                [1/np.sqrt(3), -2/np.sqrt(6), 0]])
    result_LMS=np.power(10,np.dot(trans_l_al_be2lLMS, result_l_al_be))
    trans_LMS2RGB=np.array([[4.4679, -3.5873, 0.1193],
                            [-1.2186, 2.3809, -0.1624],
                            [0.0497, -0.2439, 1.2045]])
    result_RGB=np.dot(trans_LMS2RGB,result_LMS)

    #clipping values bigger than 1 or smaller than 0 to [0,1]
    result_RGB[np.where(result_RGB<0)]=0
    result_RGB[np.where(result_RGB>1)]=1
    result_RGB=result_RGB*255
    
    #reshape to original size
    result_r=result_RGB[0,:].reshape((trt_shape[0],trt_shape[1],1))
    result_g=result_RGB[1,:].reshape((trt_shape[0],trt_shape[1],1))
    result_b=result_RGB[2,:].reshape((trt_shape[0],trt_shape[1],1))
    result=np.concatenate([result_r,result_g,result_b],axis=2).astype(np.uint8)

    return result

os.chdir('D:/Work/Color Enhancement/code/Color Transfer')
s_img=io.imread('source.png')
t_img=io.imread('target.png')
rlt=CT(s_img, t_img)
plt.figure()
plt.imshow(t_img)
plt.figure()
plt.imshow(rlt)
plt.show()