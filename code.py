import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
import cv2 as cv
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

img_name = "image1.jpg"
wm_name = "watermark2.jpg"
watermarked_img = "Watermarked_Image.jpg"
watermarked_extracted = "watermarked_extracted.jpg"
key = 50
bs = 8
w1 = 64
w2= 64
fact = 8
indx = 0
indy = 0
b_cut = 50
val1  = []
val2  = []
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
    	return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def NCC(img1, img2):
	return abs(np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2)))
def dct2(a):
	return cv.dct(a)
    #return dct(dct(a.T, norm="ortho").T, norm="ortho")
   
def idct2(a):
	return cv.idct(a)
	#return idct(idct(a.T, norm='ortho').T, norm='ortho')  

def watermark_image(img, wm):
	
	c1, c2 = np.size(img,0), np.size(img,1)
	c1x = c1
	c2x = c2
	c1-= b_cut*2
	c2-= b_cut*2
	w1, w2 = np.size(wm,0), np.size(wm, 1)
	
	print(c1, c2, w1, w2)

	if(c1*c2//(bs*bs) < w1*w2):
		print("watermark too large.")
		return img

	st = set()
	blocks = (c1//bs)*(c2//bs)
	print("Blocks availaible", blocks)
	blocks_needed = w1*w2
	
	i = 0
	j = 0
	imf = np.float32(img)
	while(i<c1x):
		while(j<c2x):
			#print(i, j)		
			dst = cv.dct(imf[i:i+bs, j:j+bs]/1.0)
			"""
			if(i==896 and j==160):
				print(dst)
				print(cv.idct(dst))
			"""
			imf[i:i+bs, j:j+bs] = cv.idct(dst)
			j+=bs
		j = 0
		i+=bs
	#print(np.size(imf))
	#print(imf[512:520, 512:520])
	final = img
	random.seed(key)
	i = 0
	print("Blocks needed", blocks_needed)
	cnt = 0
	while(i < blocks_needed):				
		to_embed = wm[i//w2][i%w2]
		ch = 0
		if(to_embed >= 127):
			to_embed = 1
			ch = 255
		else:
			to_embed = 0
		
		wm[i//w2][i%w2] = ch
		"""
		ch = 255
		new_img[i//w2][i%w2] = ch
		print(new_img[i//w2][i%w2], ch)
		"""
		#1- odd, 0 - even
		x = random.randint(1, blocks)
		#print("i",i,x)
		if(x in st):
			#print("there")
			continue
		st.add(x)
		n = c1//bs
		m = c2//bs
		#print("nmx",n,m,x)
		ind_i = (x//m)*bs + b_cut
		ind_j = (x%m)*bs + b_cut
		#print(ind_i, ind_j)
		#print(ind_i, ind_j)
		#print(imf[ind_i:ind_i+bs, ind_j:ind_j+bs])
		dct_block = cv.dct(imf[ind_i:ind_i+bs, ind_j:ind_j+bs]/1.0)
		elem = dct_block[indx][indy]
		elem /= fact
		ch = elem
		if(to_embed%2==1):	
			if(math.ceil(elem)%2==1):
				elem = math.ceil(elem)
			else:
				elem = math.ceil(elem)-1
		else:
			if(math.ceil(elem)%2==0):
				elem = math.ceil(elem)
			else:
				elem = math.ceil(elem)-1

		
		dct_block[indx][indy] = elem*fact
		#dct_block[0][0] = elem
		val1.append((elem*fact, to_embed))
		if(cnt < 5):
			#print(x, elem*fact , to_embed)
			#print(dct_block)
			cnt+=1
		
		final[ind_i:ind_i+bs, ind_j:ind_j+bs] = cv.idct(dct_block)
		imf[ind_i:ind_i+bs, ind_j:ind_j+bs] = cv.idct(dct_block)
		"""if(cnt<5):
			print(x)
			print(dct_block)
			cnt += 1
		"""
		i += 1

	#print(wm)
	final = np.uint8(final)
	#print(final[512:520, 512:520])
	#=============PSNR==========
	print("PSNR is:", psnr(imf, img))
	#=========================
	#new_img = np.uint8(new_img)
	
	#print(np.unique(new_img))
	cv.imshow("Final", final)
	cv.imwrite(watermarked_img , final)
	return imf

def extract_watermark(img, ext_name):
	c1x, c2x = np.size(img,0), np.size(img,1)
	
	if(c1x!=1000 or c2x != 1000):
		img = cv.resize(img, (1000, 1000))
		c1x = 1000
		c2x = 1000
	c1 = c1x - b_cut*2
	c2 = c2x-b_cut*2
	blocks = (c1//bs)*(c2//bs)
	blocks_needed = w1*w2

	wm = [[0 for x in range(w1)] for y in range(w2)]
	st = set()
	random.seed(key)
	i = 0
	cnt = 0
	#print("Blocks needed", blocks_needed)
	while(i<blocks_needed):
		curr = 0
		x = random.randint(1, blocks)
		if(x in st):
			#print("there")
			continue
		st.add(x)
		n = c1//bs
		m = c2//bs
		ind_i = (x//m)*bs + b_cut
		ind_j = (x%m)*bs + b_cut
		dct_block = cv.dct(img[ind_i:ind_i+bs, ind_j:ind_j+bs]/1.0)
		
		elem = dct_block[indx][indy]
		elem = math.floor(elem+0.5)
		"""
		if(cnt < 5):
			print(elem )
			cnt+=1
		"""
		elem /= fact
		
		if(elem%2 == 0):
			curr = 0
		else:
			curr = 255
		val2.append( (elem, bool(curr)))
		
		wm[i//w2][i%w2] = curr
		i+=1
		
		

	wm = np.array(wm)
	"""
	for i in range(64):
		print(wm[30][i])
	print(wm)
	"""
	cv.imwrite(ext_name , wm)
	print("Watermark extracted and saved in", ext_name)
	return wm


#======================Geometric attacks=================

##Scaling
def ScalingBigger(img):
	bigger = cv.resize(img, (1100, 1100))
	return bigger

def ScalingHalf(img):
	half = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)
	return half

##Cut attack

def Cut100Rows(img):
	new = [[0.0 for x in range(1000)] for y in range(900)]
	ni = 0
	nj = 0
	for i in range(1000):
		for j in range(1000):
			if(i<400 or i>=500):
				new[ni][nj] = img[i][j]
				nj += 1
				if(nj == 1000):
					nj = 0
					ni += 1
	new = np.array(new)
	return new


#=====================Signal Attacks=====================
def AverageFilter(img):
	kernel = np.ones((5,5),np.float32)/25
	dst = cv.filter2D(img,-1,kernel)
	return dst

def MedianFilter(img):
	m, n = img.shape 
	img_new1 = np.zeros([m, n])
	for i in range(1, m-1): 
	    for j in range(1, n-1): 
	        temp = [img[i-1, j-1], 
	               img[i-1, j], 
	               img[i-1, j + 1], 
	               img[i, j-1], 
	               img[i, j], 
	               img[i, j + 1], 
	               img[i + 1, j-1], 
	               img[i + 1, j], 
	               img[i + 1, j + 1]] 
          
        	temp = sorted(temp) 
        	img_new1[i, j]= temp[4] 
	img_new1 = img_new1.astype(np.uint8) 
	return img_new1
  
def noisy(noise_typ,image):
	if noise_typ == "gauss":
		noise = np.random.normal(0.1, 0.01 ** 0.5,image.shape)        
		noisy = image+noise
		return noisy
	elif noise_typ == "s&p":
		prob = 0.05
		output = np.zeros(image.shape,np.uint8)
		thres = 1 - prob 
		for i in range(image.shape[0]):
		    for j in range(image.shape[1]):
		        rdn = random.random()
		        if rdn < prob:
		            output[i][j] = 0
		        elif rdn > thres:
		            output[i][j] = 255
		        else:
		            output[i][j] = image[i][j]
		return output
	elif noise_typ =="speckle":
		speckle = np.random.normal(0,1,image.size)
		speckle = speckle.reshape(image.shape[0],image.shape[1]).astype('uint8')
		noisy = image + image * speckle
		return noisy



if __name__ == "__main__":

	print("Main image: " + img_name)
	print("Watermark: " + wm_name)

	print("===================================EMBEDDING WATERMARK======================")
	img =  cv.imread(img_name, 0) 
	#print(img)
	#img = np.float32(img)
	#print(img)
	#cv.imshow('Cover image', img)
	wm = cv.imread(wm_name, 0) 
	wm = cv.resize(wm, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
	#wm = np.float32(wm)
	#print(wm)
	
	cv.imshow('Watermark image', wm)
	"""
	for i in range(10):
		print(random.randint(0, 100))
	"""

	wmed = watermark_image(img, wm)
	
	print("\nWatermarking Done!\n")
	
	print("===================================EXTRACTING WATERMARK======================")
	#print(wmed[512:520, 512:520])

	wx = extract_watermark(wmed, watermarked_extracted)
	print("NCC:", NCC(wm, wx))

	#=================NCC==================
	x = cv.imread(wm_name)
	y = cv.imread(watermarked_extracted)
	cv.waitKey()
	ch = 0
	#print("NCC:",cv.matchTemplate(x,y,'cv.TM_CCORR_NORMED'))
	#======================================
	
	
	print("\n\n=============================ATTACKS=============================\n")

	print("Checking when image is scaled to half:")
	x = ScalingHalf(wmed)
	wx = extract_watermark(x, "Extracted_GeoAtt_Half.jpg")
	print("NCC:", NCC(wm, wx))

	print("\nChecking when image is scaled to 1100x1100:")
	x = ScalingBigger(wmed)
	wx = extract_watermark(x, "Extracted_GeoAtt_Bigger.jpg")
	print("NCC:", NCC(wm, wx))

	print("\nChecking when 100 rows of image are cut:")
	x = Cut100Rows(wmed)
	wx = extract_watermark(x, "Extracted_GeoAtt_Cut100Rows.jpg")
	print("NCC:", NCC(wm, wx))

	print("\nChecking when Average filter is applied:")
	x = AverageFilter(wmed)
	wx = extract_watermark(x, "Extracted_SigAtt_AvgFilter.jpg")
	print("NCC:", NCC(wm, wx))

	print("\nChecking when Median filter is applied:")
	x = MedianFilter(wmed)
	wx = extract_watermark(x, "Extracted_SigAtt_MedFilter.jpg")
	print("NCC:", NCC(wm, wx))

	print("\nChecking when Gaussian noise is added:")
	x = noisy("gauss", wmed)
	#wx = extract_watermark(x, "Extracted_SigAtt_GaussNoise.jpg")
	wx = cv.imread("Extracted_SigAtt_GaussNoise.jpg", 0)
	print("NCC:", NCC(wm, wx))

	print("\nChecking when Salt & Pepper noise is added:")
	x = noisy("s&p", wmed)
	wx = extract_watermark(x, "Extracted_SigAtt_s&pNoise.jpg")
	print("NCC:", NCC(wm, wx))
	
	print("\nChecking when speckle noise is added:")
	x = noisy("speckle", wmed)
	wx = extract_watermark(x, "Extracted_SigAtt_SpeckNoise.jpg")
	print("NCC:", NCC(wm, wx))
	