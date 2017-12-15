import numpy as np
from scipy import signal



def ModifiedSingularityIndex2D(I1, nrScales = 5, preservePolarity = 0, min_scale = 1.5):


    assert type(I1) is np.ndarray, 'Input image must be Numpy array'
    assert I1.ndim == 2, 'Input image must be 2-D'

    I1 = I1.astype(float)
    R,C = I1.shape

    scaleHistograms = np.zeros((R,C,nrScales))
    orientationHistograms = np.zeros((R,C,nrScales))

    for scale in np.arange(nrScales,0,-1):

        largest_scale = min_scale * ((np.sqrt(2)**(scale-1)))
        sigmabig = np.ceil(np.sqrt((1 - (0.2)**2) / (0.2**2)) * largest_scale)
        siz = np.round(sigmabig * 6) + 1
        if siz%2 == 0: siz += 1

        gbig = matlab_style_gauss2D(shape=(siz,siz), sigma = sigmabig)
        mu = signal.convolve2d(I1, gbig, boundary='symm', mode='same')
        I = I1 - mu

        sigma = min_scale * (np.sqrt(2)**(scale-1))
        siz = np.round(sigma * 4) + 1
        if siz%2 == 0: siz += 1

        X,Y = np.meshgrid(np.arange(-siz,siz+1), np.arange(-siz,siz+1))

        theta1 = 0
        theta2 = np.pi / 3.
        theta3 = 2 * np.pi / 3.

        u = X * np.cos(theta1) - Y * np.sin(theta1)
        v = X * np.sin(theta1) + Y * np.cos(theta1)

        # This is the mother Gaussian.

        G0 = (1. / (2 * np.pi * sigma**2)) * np.exp(-(u**2 + v**2) / (2. * sigma** 2))

        # Since the isotropic Gaussian is rotationally symmetric, all derivatives
        # are defined in terms of this G0.

        G20 = G0 * ((u**2 / sigma**4) - (1. / sigma**2)) #Second partial derivative along u

        u = X * np.cos(theta2) - Y * np.sin(theta2)
        G260 = G0 * ((u**2 / sigma**4) - (1. / sigma**2)) #Second partial derivative along u

        u = X * np.cos(theta3) - Y * np.sin(theta3)
        G2120 = G0 * ((u**2 / sigma**4) - (1. / sigma**2)) #Second partial derivative along u

        G20 = G20 - G20.mean()
        G260 = G260 - G260.mean()
        G2120 = G2120 - G2120.mean()

        J20 = signal.convolve2d(I, G20, boundary='symm', mode='same')
        J260 = signal.convolve2d(I, G260, boundary='symm', mode='same')
        J2120 = signal.convolve2d(I, G2120, boundary='symm', mode='same')

        Nr = (2 * np.sqrt(3) / 9.) * (J260**2 - J2120**2 + J20*J260 - J20*J2120)

        Dr = (2. / 9.) * (2 * J20**2 - J260**2 - J2120**2
                          + J20*J260 - 2*J260*J2120 + J20*J2120)

        # The implementation below is for atan2 from wiki
        angles = np.zeros(I.shape)

        indx = Dr > 0
        angles[indx] = np.arctan(Nr[indx] / Dr[indx])

        indx = (Nr >= 0) & (Dr < 0)
        angles[indx] = np.arctan(Nr[indx] / Dr[indx]) + np.pi

        indx = (Nr < 0) & (Dr < 0)
        angles[indx] = np.arctan(Nr[indx] / Dr[indx]) - np.pi

        indx = (Nr > 0) & (Dr == 0)
        angles[indx] = np.pi / 2

        indx = (Nr < 0) & (Dr == 0)
        angles[indx] = -np.pi / 2

        angles = 0.5 * angles
        sigma1 = 1.7754 * sigma

        siz = np.round(sigma1 * 4) + 1
        if siz%2 == 0: siz += 1

        X,Y = np.meshgrid(np.arange(-siz,siz+1), np.arange(-siz,siz+1))
        theta1 = 0

        u = X * np.cos(theta1) - Y * np.sin(theta1)
        v = X * np.sin(theta1) + Y * np.cos(theta1)

        G0_half = (1. / (2 * np.pi * sigma1**2)) * np.exp(-(u**2 + v**2) / (2. * sigma** 2))
        G0u = -(1./sigma1)**2 * u * G0_half # First partial derivative along u

        theta1 = np.pi / 2
        u = X * np.cos(theta1) - Y * np.sin(theta1)
        G90u = -(1./sigma1)**2 * u * G0_half # First partial derivative along u

        G0u = G0u - G0u.mean()
        G90u = G90u - G90u.mean()

        J0u = signal.convolve2d(I, G0u, boundary='symm', mode='same')
        J90u = signal.convolve2d(I, G90u, boundary='symm', mode='same')

        # J2 is the second order derivative along the direction specified by angles
        J2 = ((1./3.) * (1 + (2 * np.cos(2 * angles))) * J20
             + ((1. / 3) * (1 - np.cos(2 * angles) + (np.sqrt(3) * np.sin(2 * angles))) * J260)
             + ((1. / 3) * (1 - np.cos(2 * angles) - (np.sqrt(3) * np.sin(2 * angles))) * J2120))

        # J1 is the first order derivative along the direction specified by angles

        J1 = (J0u * np.cos(angles)) + (J90u * np.sin(angles))

        if G0.sum() != 0:
            G0 = G0 / G0.sum()

        J = signal.convolve2d(I, G0, boundary='symm', mode='same')

        if not preservePolarity:
            psi_scale = sigma**2 * (np.abs(J * J2) / (1 + (np.abs(J1) ** 2)))
        else:
            psi_scale = sigma**2 * (np.abs(J) * J2 / (1 + (np.abs(J1) ** 2)))

        if scale == nrScales:

            if preservePolarity:

                # suppress negative response (experimental)
                polarity = np.sign(psi_scale)
                negIdx = polarity > 0
                psi_scale[negIdx] = 0

            psi = psi_scale.copy()
            orient = angles.copy()
            scaleMap = sigma * np.abs(psi_scale)
            scaleMapIdx = scale * np.abs(psi_scale)
            psi_sum = np.abs(psi_scale)

        else:
            if not preservePolarity:
                idx = psi_scale > psi
            else:
                polarity_scale = np.sign(psi_scale)

                #suppress negative response (experimental)
                negIdx = polarity_scale > 0
                psi_scale[negIdx] = 0

                idx = (polarity_scale * psi_scale) > (polarity * psi)
                polarity[idx] = polarity_scale[idx]


            psi[idx] = psi_scale[idx]

            sigmaT = min_scale
            siz = np.round(sigmaT * 6) + 1
            if siz%2 == 0: siz += 1


            psi_g = matlab_style_gauss2D(shape=(siz,siz), sigma = sigmaT)
            psi = signal.convolve2d(psi, G0, boundary='symm', mode='same')


            orient[idx] = angles[idx]
            scaleMap = scaleMap + sigma * np.abs(psi_scale)
            scaleMapIdx = scaleMapIdx + scale * np.abs(psi_scale)
            psi_sum = psi_sum + np.abs(psi_scale)


        orientationHistograms[:,:,scale-1] = angles
        scaleHistograms[:,:,scale-1] = np.abs(psi_scale)



    scaleMap = scaleMap / (psi_sum + 1e-10)
    scaleMapIdx = scaleMapIdx / (psi_sum + 1e-10)

    orientation = orient.copy()
    nms_orient = orient.copy()

    # Simple NMS implementation
    nms_orient = nms_orient * 180 / np.pi
    idx = nms_orient < 0
    nms_orient[idx] += 180

    Q = np.zeros((R,C))
    idx = ((nms_orient >= 0) & (nms_orient <= 22.5)) | ((nms_orient >= 157.5) & (nms_orient <= 180))
    Q[idx] = 0 # suppress in east-west

    idx = (nms_orient > 22.5) & (nms_orient <= 67.5)
    Q[idx] = 1 # suppress in north east, south west

    idx = (nms_orient > 67.5) & (nms_orient <= 112.5)
    Q[idx] = 2 # supppress in north south

    idx = (nms_orient > 112.5) & (nms_orient <= 157.5)
    Q[idx] = 3 # supppress in north west south east

    pos_psi, neg_psi, posNMS, negNMS = np.zeros((4,R,C))

    if not preservePolarity:

        NMS = psi.copy()

        for i in range(1,R-1):
            for j in range(1,C-1):

                if Q[i,j] == 0:
                    if ((psi[i,j] <= psi[i,j-1]) or
                        (psi[i,j] <= psi[i,j+1])):
                        NMS[i,j] = 0

                elif Q[i,j] == 1:
                    if ((psi[i,j] <= psi[i-1,j+1]) or
                        (psi[i,j] <= psi[i+1,j-1])):
                        NMS[i,j] = 0

                elif Q[i,j] == 2:
                    if ((psi[i,j] <= psi[i-1,j]) or
                        (psi[i,j] <= psi[i+1,j])):
                        NMS[i,j] = 0

                elif Q[i,j] == 3:
                    if ((psi[i,j] <= psi[i-1,j-1]) or
                        (psi[i,j] <= psi[i+1,j+1])):
                        NMS[i,j] = 0

    else:

        K_temp = polarity * psi
        NMS = K_temp.copy()

        for i in range(1,R-1):
            for j in range(1,C-1):

                if Q[i,j] == 0:
                    if ((K_temp[i,j] <= K_temp[i,j-1]) or
                        (K_temp[i,j] <= K_temp[i,j+1])):
                        NMS[i,j] = 0

                elif Q[i,j] == 1:
                    if ((K_temp[i,j] <= K_temp[i-1,j+1]) or
                        (K_temp[i,j] <= K_temp[i+1,j-1])):
                        NMS[i,j] = 0

                elif Q[i,j] == 2:
                    if ((K_temp[i,j] <= K_temp[i-1,j]) or
                        (K_temp[i,j] <= K_temp[i+1,j])):
                        NMS[i,j] = 0

                elif Q[i,j] == 3:
                    if ((K_temp[i,j] <= K_temp[i-1,j-1]) or
                        (K_temp[i,j] <= K_temp[i+1,j+1])):
                        NMS[i,j] = 0



        idx = polarity != -1 # Negative going impulses have positive polarity
        neg_psi[idx] = psi[idx]
        negNMS[idx] = NMS[idx]

        idx = polarity == -1 # Positive going impulses have negative polarity
        pos_psi[idx] = -psi[idx]
        posNMS[idx] = NMS[idx]


    SI = {}

    SI['orientation'] = orientation
    SI['orientationHistograms'] = orientationHistograms
    SI['NMS'] = NMS
    SI['scaleMap'] = scaleMap
    SI['scaleHistograms'] = scaleHistograms
    SI['pos_psi'] = pos_psi
    SI['posNMS'] = posNMS
    SI['neg_psi'] = neg_psi
    SI['negNMS'] = negNMS
    
    
    return SI
    
    
    
    
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h