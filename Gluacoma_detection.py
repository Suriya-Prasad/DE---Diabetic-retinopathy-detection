import cv2
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

def openimage(i):
    t1 = i // 10
    t2 = i % 10
    filename = "HRF/Images/image0"+str(t1)+str(t2)+".jpg" #for initial images

    return filename

def saveimage(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Preprocessed/Images/image_0"+str(t1)+str(t2)+".jpg" #for initial images

    return filename

def openimagep(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Preprocessed/Images/image_0"+str(t1)+str(t2)+".jpg" #for initial images

    return filename

def saveimagep(i):
    t1 = i // 10
    t2 = i % 10
    filename = "Output/Images/image_0"+str(t1)+str(t2)+".jpg" #for initial images

    return filename

def solution_set(npop,shape):
    height,width = shape[0],shape[1]
    xmin,ymin = 50,100
    xmax,ymax = width-50,height-100
    random_points = []
    b1 = random.sample(range(ymin,ymax), npop)
    b2 = random.sample(range(xmin,xmax), npop)

    return list(zip(b1,b2))

def fitness_value(img,point):
    x,y = point[1],point[0]
    radius = 30

    blank = np.zeros(img.shape,dtype='uint8')

    cv2.circle(blank,(x,y),radius,(255,0,0),-1)
    pixels = []
    for i in range(len(blank)):
        for j in range(len(blank[0])):
            if blank[i,j] == 255:pixels.append([i,j])
    
    min_intensity = None
    for i in pixels:
        if min_intensity is None or img[i[0],i[1]] < min_intensity:
            min_intensity = img[i[0],i[1]]
    
    img = cv2.bitwise_and(img,img,mask=blank)

    ret, threshold = cv2.threshold(img, min_intensity+1, 1, cv2.THRESH_BINARY)

    s = 0
    for i in threshold:
        s += sum(i)

    z=(((100*2)**2)*255)
    y=1000*(1-(s/z))

    return y

def opticdecross( population, mutant, cross, shape ):
    height, width = shape[0], shape[1]
    xmin, ymin = 50,100
    xmax, ymax = width-50, height-100
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

# CLAHE and Median Blur
def preprocessing():
    for i in tqdm(range(1,21)):
        open_path = openimage(i)
        img = cv2.imread(open_path)
        if img is None:
            print(f"image {i} not found")

        dimension = img.shape

        # img = cv2.resize(img,(300,300),interpolation=cv2.INTER_AREA)

        b,g,r = cv2.split(img)
        img = g

        blank = np.zeros(img.shape,dtype='uint8')

        clahe = cv2.createCLAHE(clipLimit=2)
        clahe_image = clahe.apply(img)

        blur = cv2.medianBlur(clahe_image,55)
        
        save_path = saveimage(i)
        cv2.imwrite(save_path,blur)

def point_computation():
    for i in range(1,2):
        # open_path = openimagep(i)
        # img = cv2.imread(open_path)
        img = cv2.imread('WorkingProject\Preprocessed\Images\image_001.jpg')
        img,g,r = cv2.split(img)
        
        npop,ngen = 20,50

        if img is None:
            print(f"image {i} is not found")
        
        populations = [solution_set(npop,img.shape)]

        fitness = {}

        for j in range(npop):
            if populations[0][j] not in fitness:
                fitness[populations[0][j]] = fitness_value(img,populations[0][j])
        print(len(fitness))
        
        a,f,cross = [_ for _ in range(npop)],0.4,0.7

        fit_values = np.zeros([ngen,npop])    

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

                cross_point = opticdecross(populations[k][m],mutant,cross,img.shape)

                if cross_point not in fitness:
                    q1 = fitness_value(img,cross_point)
                    fitness[cross_point] = q1
                q2 = fitness[populations[k][m]]

                if j == len(populations):
                    populations.append([])

                if q1 <= q2:
                    populations[j].append(cross_point)
                    # fit_values[j][m] = q1
                else:
                    populations[j].append(populations[k][m])
                    # fit_values[j][m] = q2
                
            # fit_values[j] = sorted(fit_values[j],reverse=True)
            # print(fit_values[j])
            print(len(fitness))
        print(fitness)

point_computation()