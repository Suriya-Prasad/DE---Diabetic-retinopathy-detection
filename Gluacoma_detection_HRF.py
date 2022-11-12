import cv2
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import openpyxl
import math


def openimage(i):
    t1 = i // 10
    t2 = i % 10
    filename = "HRF Dataset/Images/image_0"+str(t1)+str(t2)+".jpg"

    return filename


def saveimage(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Preprocessed_HRF/Images/image_0"+str(t1)+str(t2)+".jpg"

    return filename


def openimagep(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Preprocessed_HRF/Images/image_0"+str(t1)+str(t2)+".jpg"

    return filename


def saveimagep(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Output_HRF/Images/image_0"+str(t1)+str(t2)+".jpg"

    return filename


def openimageo(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Output_HRF/Images/image_0"+str(t1)+str(t2)+".jpg"

    return filename


def saveimageo(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Truth Values_HRF/Images/image_0"+str(t1)+str(t2)+".jpg"

    return filename


# # Original image Histogram

# img = cv2.imread('Drive Dataset/Images/image_001.jpg')
# if img is None:
#     print("image not found")

# # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# b,g,r = cv2.split(img)

# hist = cv2.calcHist([b],[0],None,[256],[0,255])
# plt.plot(hist)
# hist = cv2.calcHist([g],[0],None,[256],[0,255])
# plt.plot(hist)
# hist = cv2.calcHist([r],[0],None,[256],[0,255])
# plt.plot(hist)

# hist = cv2.calcHist([img],[0],None,[256],[0,256])

# cv2.imshow('original',img)
# cv2.waitKey(0)
# plt.plot(hist)
# plt.show()


# # Global AHE

# img = cv2.imread('Drive Dataset/Images/image_001.jpg')
# if img is None:
#     print("image not found")

# img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(img_hsv)
# v = cv2.equalizeHist(v)
# merged_hsv = cv2.merge((h,s,v))
# bgr_enhanced = cv2.cvtColor(merged_hsv,cv2.COLOR_HSV2BGR)
# b,g,r = cv2.split(bgr_enhanced)

# hist = cv2.calcHist([b],[0],None,[256],[0,255])
# plt.plot(hist)
# hist = cv2.calcHist([g],[0],None,[256],[0,255])
# plt.plot(hist)
# hist = cv2.calcHist([r],[0],None,[256],[0,255])
# plt.plot(hist)

# cv2.imshow('globalAHE',bgr_enhanced)
# cv2.waitKey(0)

# plt.show()


# CLAHE and Median Blur

def preprocessing():
    for i in tqdm(range(1,46),desc='Pre-Processing in progress'):
        open_path = openimage(i)
        img = cv2.imread(open_path)

        if img is None:
            print(f"image {i} not found")

        dimension = img.shape

        b,g,r = cv2.split(img)
        img = g

        blank = np.zeros(img.shape,dtype='uint8')

        clahe = cv2.createCLAHE(clipLimit=2)
        clahe_image = clahe.apply(img)

        blur = cv2.medianBlur(clahe_image,145)
        
        save_path = saveimage(i)
        cv2.imwrite(save_path,blur)


def solution_set(npop,shape):
    height,width = shape[0],shape[1]
    xmin,ymin = width//10,height//3
    xmax,ymax = width-xmin-1,height-ymin-1
    random_points = []
    b1 = random.sample(range(ymin,ymax), npop)
    b2 = random.sample(range(xmin,xmax), npop)
    
    return list(zip(b1,b2))


def fitness_value(img,point):
    x,y = point[1],point[0]
    radius = img.shape[0]//13

    img = img[y-radius:y+radius+1,x-radius:x+radius+1]
    s = img.sum()
    y = 1 - (s / 1000)
    
    return y


def opticdecross( population, mutant, cross, shape ):
    height, width = shape[0], shape[1]
    xmin, ymin = width//10,height//3
    xmax, ymax = width-xmin,height-ymin
    ir = random.randrange(0,2)
    cross_point = [0,0]
    for i in range(2):
        if( random.uniform(0,1) <= cross or i==ir ):
            if i == 0:
                if( mutant[0] <= ymax and mutant[0] >= ymin ):
                    cross_point[0] = mutant[0]
                else:
                    cross_point[0] = population[0]
            if i == 1:
                if( mutant[1] <= xmax and mutant[1] >= xmin ):
                    cross_point[1] = mutant[1]
                else:
                    cross_point[1] = population[1]
    return tuple(cross_point)


def check_bounds(mutant,shape):
    if shape[0]//3 <= mutant[0] <= shape[0]-shape[0]//3 and 50 <= mutant[1] <= shape[1]-shape[1]//10:
        return mutant
    elif shape[0]//3 <= mutant[0] <= shape[0]-shape[0]//3:
        return (mutant[0],random.randint(shape[1]//10,shape[1]-(shape[1]//10)-1))
    else:
        return (random.randint(shape[0]//3,shape[0]-(shape[0]//3)-1),mutant[1])


def segmentation():

    workbook = openpyxl.load_workbook("Retinal fundus images centres.xlsx")
    sheet = workbook['Sheet6']

    for i in tqdm(range(1,46),desc='Image Segmentation in progress'):

        open_path = openimagep(i)
        img = cv2.imread(open_path)

        original_image = img
        b,img,r = cv2.split(img)

        npop,ngen = 80,350

        if img is None:
            print(f"image {i} is not found")
        
        populations = [solution_set(npop,img.shape)]

        fitness = {}

        for j in range(npop):
            if populations[0][j] not in fitness:
                fitness[populations[0][j]] = fitness_value(img,populations[0][j])
        
        a,f,cross = [_ for _ in range(npop)],0.6,0.7

        min_point,min_fitness = None,None

        for j in range(ngen):
            if j == 0:k = 0
            else:k = j-1

            for m in range(npop):
                b = random.sample(range(npop),npop)
                c = [_ for _ in b if _ != m] 
                r1 = a[b[c[0]]]
                r2 = a[b[c[1]]]
                r3 = a[b[c[2]]]

                mutant = [int(abs(populations[k][r1][0] + f*(-(populations[k][r2][0])-populations[k][r3][0]))),int(abs(populations[k][r1][1] + f*(-(populations[k][r2][1])-populations[k][r3][1])))]

                mutant = check_bounds(mutant,img.shape)

                cross_point = opticdecross(populations[k][m],mutant,cross,img.shape)

                if cross_point not in fitness:
                    q1 = fitness_value(img,cross_point)
                    fitness[cross_point] = q1
                q2 = fitness[populations[k][m]]

                if j == len(populations):
                    populations.append([])

                if q1 <= q2:
                    populations[j].append(cross_point)
                    if min_fitness is None or min_fitness >= q1:
                        min_fitness,min_point = q1,cross_point

                else:
                    populations[j].append(populations[k][m])
                    if min_fitness is None or min_fitness >= q2:
                        min_fitness,min_point = q2,populations[k][m]

        cv2.circle(original_image,(min_point[1],min_point[0]),original_image.shape[0]//13,(0,0,255),5)
        save_path = saveimagep(i)
        cv2.imwrite(save_path,original_image)

        distance = ((sheet['D'+str(i+3)].value - min_point[1])**2 + (sheet['E'+str(i+3)].value - min_point[0])**2)**0.5

        sheet['B'+str(i+3)].value = min_point[1]
        sheet['C'+str(i+3)].value = min_point[0]
        sheet['F'+str(i+3)].value = distance

        sheet['G'+str(i+3)].value = 'N' if distance > original_image.shape[0]//13 else 'Y'
    
    workbook.save("Retinal fundus images centres.xlsx")



def truth_values():

    workbook = openpyxl.load_workbook("Retinal fundus images centres.xlsx")
    sheet = workbook['Sheet6']
    
    for i in tqdm(range(1,46),desc='Truth-Value in progress'):
        open_path = openimageo(i)
        img = cv2.imread(open_path)

        cv2.circle(img,(sheet['D'+str(i+3)].value,sheet['E'+str(i+3)].value),img.shape[0]//13,(0,255,0),5)
        save_path = saveimageo(i)
        cv2.imwrite(save_path,img)

        blank1 = np.zeros(img.shape,dtype='uint8')
        blank2 = np.zeros(img.shape,dtype='uint8') 

        cv2.circle(blank1,(sheet['D'+str(i+3)].value,sheet['E'+str(i+3)].value),img.shape[0]//13,(0,255,0),-1)
        cv2.circle(blank2,(sheet['B'+str(i+3)].value,sheet['C'+str(i+3)].value),img.shape[0]//13,(0,255,0),-1)

        and_image = cv2.bitwise_and(blank1,blank2,mask=None)
        xor_image_actual = cv2.bitwise_xor(blank1,and_image,mask=None)
        xor_image_predicted = cv2.bitwise_xor(blank2,and_image,mask=None)

        b,and_image,r = cv2.split(and_image)
        b,xor_image_actual,r = cv2.split(xor_image_actual)
        b,xor_image_predicted,r = cv2.split(xor_image_predicted)

        sheet['H'+str(i+3)].value = and_image.sum()//255
        sheet['I'+str(i+3)].value = xor_image_actual.sum()//255
        sheet['J'+str(i+3)].value = xor_image_predicted.sum()//255
    
    workbook.save("Retinal fundus images centres.xlsx")



preprocessing()
segmentation()
truth_values()