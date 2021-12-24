import operator
import numpy as np
import cv2

class Ex_not_recognize(Exception):
    def __init__(self):
        self.text="Sudoku was not recognized!!!"

class Recognize_Sudoku():
    def __init__(self,DX=15,DY=15,n_count_max=48,n_count_min=28,maxLineLength = 650,
                 minLineLength = 280,maxLineGap = 20,dtheta=np.pi/400,drho=1,level=10,out_size=28):
        self.DX=DX #Max distance (in pixels) between 2 vertical lines. If distance between 2 lines less DX than 2 lines are counted as one line
        self.DY=DY #Max distance (in pixels) between 2 horizontal lines. If distance between 2 lines less DY than 2 lines are counted as one line
        self.n_count_max=n_count_max # Maximum number of line crossings with other lines. It used to whether a line belongs to sudoku
        self.n_count_min=n_count_min # Minimum number of line crossings with other lines. It used to whether a line belongs to sudoku

        self.SIZE=10 # Number of vertical / number of horizontal lines in sudoku - const
        self.maxLineLength = maxLineLength #The maximum length (pixels) of a line that can belong to the sudoku

        #This parameters used to Hough Transform (library cv2)
        self.minLineLength = minLineLength #The minimum length of a line
        self.maxLineGap = maxLineGap #Max gap between lines при котором они все еще считаются одной линией
        self.dtheta=dtheta #Resolution for angle
        self.drho=drho#Resolution for distance
        self.level=level#Accumulator threshold parameter. Only those lines are returned that get enough votes

        self.out_size=out_size#Resolution to which we convert the sudoku recognizing digits for pixel-by-pixel comparison

        self.arr_digits=[]#List of images, reference digits for pixel-by-pixel comparison
        for ii in range(1,self.SIZE):
            #Load digits images
            #With these images we will compare the numbers that need to be recognized pixel by pixel.
            img=cv2.imread(str(ii)+'.png')
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
            self.arr_digits.append(thresh)

    def find_h_lines(self,lines,edges_image):
        '''
            @param lines - лист всех распознанных линий!!!!!!!!!!!!!!!
            @param edges_image - двухцветное изображение на котором распознаем судоку'''
        h_lines=[] #List with horizontal lines for intermediate calculation
        out=[]  # Output list, contains horizontal lines
        #h_lines like [x1,y1,x2,y2]

        for ii in range(len(lines)):
            x1=int(lines[ii][0][0]) #Coordinates of the beginning of the horizontal line
            x2=int(lines[ii][0][2]) #ending

            y_u=int(lines[ii][0][1]-2) # To determine the number of intersections,
            y_d=int(lines[ii][0][1]+2)# select two parallel lines shifted up and down by 2 pixels

            # чтобы определить принадлежит ли линия судоку считаем число переходов с черное на белое вдоль линии
            #сдвинутой вверх на 2 пикселя
            prev_pixel=0
            count_u=0
            if y_u>=0:
                for jj in edges_image[y_u,x1:x2]:
                    if jj!=prev_pixel:
                        count_u+=1
                    prev_pixel=jj

            #то же для линии сдвинутой вниз
            prev_pixel=0
            count_d=0
            if y_d<edges_image.shape[0]:
                for jj in edges_image[y_d,x1:x2]:
                    if jj!=prev_pixel:
                        count_d+=1
                    prev_pixel=jj

            expression=count_u>=self.n_count_min or count_d>=self.n_count_min  #число переходов с черное на белое хотя бы для одой линии должно быть не меньше n_count_min
            expression=expression and count_d<=self.n_count_max and count_u<=self.n_count_max #число переходов с черное на белое обязательно для 2 линий должно быть меньше чем n_count_max
            # чтобы отсечь линии не связанные с судоку
            expression = expression and lines[ii][0][1]==lines[ii][0][3] #проверка является ли линия горизонтальной
            expression=expression and ((lines[ii][0][0]-lines[ii][0][2])**2+(lines[ii][0][1]-lines[ii][0][3])**2)**0.5<=self.maxLineLength  #Проверка на длину линии
            if expression:
                h_lines.append(lines[ii][0])
        h_lines=sorted(h_lines,key=operator.itemgetter(1)) # сортируем линии так чтобы они возрастали по координате y

        #возможна ситуация когда толстая линия распознается как несколько рядом лежащих линий их нужно отсортировать
        ii=0
        while (ii<len(h_lines)-1): #по всем линиям кроме последней
            if abs(h_lines[ii][1]-h_lines[ii+1][1])<=self.DY:
                #если соседние линии различаются меньше чем на DY то считаем их одной линией mid_lines
                mid_lines=h_lines[ii]
                mid_lines[1]=h_lines[ii][1]/2+h_lines[ii+1][1]/2 # "y" координаты mid_lines заменяем средними значениями
                mid_lines[3]=h_lines[ii][3]/2+h_lines[ii+1][3]/2
                out.append(mid_lines)
                ii+=2  # если объединили 2 линии в одну то следующую линию уже не рассматриваем
            else:
                out.append(h_lines[ii])
                ii+=1
        #для последней линии
        if abs(h_lines[-1][1]-h_lines[-2][1])<=self.DY:
            mid_lines=h_lines[-1]
            mid_lines[1]=h_lines[-1][1]/2+h_lines[-2][1]/2
            mid_lines[3]=h_lines[-1][3]/2+h_lines[-2][3]/2
            out.append(mid_lines)
        else:
            out.append(h_lines[-1])
        return out

    def find_v_lines(self,lines,edges_image):
        #lines - лист всех распознанных линий!!!!!!!!!!!!!!
        #edges_image - двухцветное изображение на котором распознаем судоку
        v_lines=[]  #лист с вертикальными линиями для промежуточных расчетов
        out=[]  # выходной лист с вертикальными линиями
        #v_lines и out имеют формат [x1,y1,x2,y2]

        for ii in range(len(lines)):
            y1=int(lines[ii][0][1])  #начало вертикальной линии
            y2=int(lines[ii][0][3])  #конец

            x_l=int(lines[ii][0][0]-2)  # чтобы определить число пересечений выберем
            x_r=int(lines[ii][0][0]+2)  # две парралельные сдвинутые влево и вправо на 2 пикселя линии

            # чтобы определить принадлежит ли линия судоку считаем число переходов с черное на белое вдоль линии
            #сдвинутой влево на 2 пикселя
            prev_pixel=0
            count_l=0
            if x_l>=0:
                for jj in edges_image[y2:y1,x_l]:
                    if jj!=prev_pixel:
                        count_l+=1
                    prev_pixel=jj

            #то же для линии сдвинутой вправо
            prev_pixel=0
            count_r=0
            if x_r<edges_image.shape[1]:
                for jj in edges_image[y2:y1,x_r]:
                    if jj!=prev_pixel:
                        count_r+=1
                    prev_pixel=jj

            # чтобы отсечь линии не связанные с судоку
            expression=count_l>=self.n_count_min or count_r>=self.n_count_min  #число переходов с черное на белое хотя бы для одой линии должно быть не меньше n_count_min
            expression=expression and count_l<=self.n_count_max and count_r<=self.n_count_max  #число переходов с черное на белое обязательно для 2 линий должно быть меньше чем n_count_max
            expression=expression and lines[ii][0][0]==lines[ii][0][2]  #Сheck the line is vertical
            expression=expression and ((lines[ii][0][0]-lines[ii][0][2])**2+(lines[ii][0][1]-lines[ii][0][3])**2)**0.5<=self.maxLineLength # Check for max line length
            if expression:
                v_lines.append(lines[ii][0])
        v_lines=sorted(v_lines,key=operator.itemgetter(0))  # Sort the lines so that their "x" coordinates increase

        #возможна ситуация когда толстая линия распознается как несколько рядом лежащих линий их нужно отфильтровать
        ii=0
        while (ii<len(v_lines)-1):  #For each lines except the last
            if abs(v_lines[ii][0]-v_lines[ii+1][0])<=self.DX:
                #если соседние линии различаются меньше чем на DX то считаем из одной линией mid_lines
                mid_lines=v_lines[ii]
                mid_lines[0]=v_lines[ii][0]/2+v_lines[ii+1][0]/2  # x координаты mid_lines заменяем средними значениями
                mid_lines[2]=v_lines[ii][2]/2+v_lines[ii+1][2]/2
                out.append(mid_lines)
                ii+=2  # If 2 lines are combined into one, then the next line is not considered
            else:
                out.append(v_lines[ii])
                ii+=1
        #For the last line
        if abs(v_lines[-1][0]-v_lines[-2][0])<=self.DX:
            mid_lines=v_lines[-1]
            mid_lines[0]=v_lines[-1][0]/2+v_lines[-2][0]/2
            mid_lines[2]=v_lines[-1][2]/2+v_lines[-2][2]/2
            out.append(mid_lines)
        else:
            out.append(v_lines[-1])
        return out

    def get_coordinates(self,h_lines,v_lines):
        # The function determines the coordinates of the intersections of vertical and horizontal lines
        #Output!!!!!!!!!!!!!
        coordx=np.zeros((self.SIZE,self.SIZE),dtype=int)
        coordy=np.zeros((self.SIZE,self.SIZE),dtype=int)
        # coordx and coordy lists have a structure similar to the "np.meshgrid" function's output arrays
        for ii in range(self.SIZE):
            for jj in range(self.SIZE):
                coordx[ii,jj]=v_lines[jj][0]
                coordy[ii,jj]=h_lines[ii][1]
        return coordx,coordy

    def compare_digit(self,in_img):
        #Recognition of a digit in the input image in_img
        #Recognition occurs by minimizing the difference between the input digit image and the reference digit image
        #Output!!!!!!!!!!!
        diff=np.zeros(self.SIZE-1,dtype=float)
        for num in range(self.SIZE-1): #For each digits from 0 to 9
            for ii in range(self.out_size):
                for jj in range(self.out_size):
                    #Calculate the sum of the squares of the difference of each pixel in the image
                    diff[num]+=(self.arr_digits[num][ii,jj]/self.out_size-in_img[ii,jj]/self.out_size)**2
        return min(enumerate(diff), key=operator.itemgetter(1))[0]+1

    def recognize_lines(self,gray):
        #Main function for recognizing sudoku lines
        edges=cv2.Canny(gray,threshold1=90,threshold2=350,apertureSize=3) ###переводим в двухцветное для распознование линий!!
        #Recognize all lines, lines list have format [[[x1,y1,x2,y2]],...[[x1,y1,x2,y2]],...]
        lines=cv2.HoughLinesP(edges,rho=self.drho,theta=self.dtheta,threshold=self.level,minLineLength=self.minLineLength,maxLineGap=self.maxLineGap)

        #Check if enough lines were recognized
        try:
            if len(lines)<2*self.SIZE:
                raise Ex_not_recognize
        except TypeError:
            raise Ex_not_recognize

        try:
            h_lines=self.find_h_lines(lines,edges) #From lines select horizontal lines that belong to sudoku
            v_lines=self.find_v_lines(lines,edges) #From lines select vertical lines that belong to sudoku
        except IndexError:
            raise Ex_not_recognize

        #Check if enough lines were recognized
        if len(h_lines)<self.SIZE or len(v_lines)<self.SIZE:
            raise Ex_not_recognize

        coordx,coordy=self.get_coordinates(h_lines,v_lines)
        return coordx,coordy,h_lines,v_lines

    def recognize_contours(self,gray,coordx,coordy):
        #!!!!!!!!!!!!!!!
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)####по другому переводим в двухцветное изображение для распознования цифр!!!
        img_erode = cv2.erode(thresh, np.ones((2, 2), np.uint8), iterations=1)

        #Output list with images of numbers that need to be recognized
        arr_recognizes_digits=[]
        for k in range(self.SIZE-1):
            arr_recognizes_digits.append([None]*(self.SIZE-1))

        for ii in range(self.SIZE-1): #For each Sudoku cell
            for jj in range(self.SIZE-1):
                # Sudoku cell coordinates
                x1=coordx[ii,jj]
                x2=coordx[ii+1,jj+1]
                y1=coordy[ii,jj]
                y2=coordy[ii+1,jj+1]
                # Recognize the contours around the numbers, pre-select the recognition area in the form of a Sudoku square
                contours,hierarchy=cv2.findContours(thresh[y1:y2,x1:x2],cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

                if len(contours)!=1:#####!!!!!!!!!!!!!!!!!!
                    for idx, contour in enumerate(contours):
                        (x_rec,y_rec,w,h)=cv2.boundingRect(contour) #Get the coordinates of the rectangle around the digit
                        if hierarchy[0][idx][3] == 0:######!!!!!!!!!!!!!!
                            #Select the area around the number, this image still needs to be brought to resolution  out_size x out_size
                            dig_crop = thresh[y1+y_rec:y1+y_rec + h, x1+x_rec:x1+x_rec + w]
                            #Convert the image to square with a resolution size_max x size_max
                            size_max = max(w, h)
                            dig_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                            if w > h:
                                # Enlarge image top-bottom (fill white)
                                y_pos = size_max // 2 - h // 2
                                dig_square[y_pos:y_pos + h, 0:w] = dig_crop
                            elif w < h:
                                # Enlarge image left-right (fill white)
                                x_pos = size_max // 2 - w // 2
                                dig_square[0:h, x_pos:x_pos + w] = dig_crop
                            else:
                                dig_square = dig_crop
                            # Resize digit to out_size x out_size
                            arr_recognizes_digits[ii][jj]=cv2.resize(dig_square, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)

        return arr_recognizes_digits

    def recognize_sudoku(self,screen):
        #функция распознания судоку
        out_puzzle=np.zeros((self.SIZE-1,self.SIZE-1))  # Output array with recognized digits, 0 corresponds to no digit
        gray=cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)

        try:
            coordx,coordy,h_lines,v_lines=self.recognize_lines(gray)
        except Ex_not_recognize as expt:
            print(expt.text)
            return None,None,None

        #Plot the lines that we recognized, for control
        img=np.array(screen)
        for l in v_lines:
            for x1,y1,x2,y2 in [l]:
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        for l in h_lines:
            for x1,y1,x2,y2 in [l]:
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("output",img)  #Input image with recognized lines
        ###
        recognizes_digits=self.recognize_contours(gray,coordx,coordy)
        for ii in range(self.SIZE-1):
            for jj in range(self.SIZE-1):
                try:
                    out_puzzle[ii,jj]=self.compare_digit(recognizes_digits[ii][jj])
                except TypeError:
                    pass
        return out_puzzle,coordx,coordy

    def save_reference_digit(self,screen):
        #This function is needed to save the reference digits for each specific sudoku
        gray=cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        try:
            coordx,coordy,_,_=self.recognize_lines(gray)
        except Ex_not_recognize as expt:
            print(expt.text)
            return None

        recognizes_digits=self.recognize_contours(gray,coordx,coordy)
        for ii in range(self.SIZE-1):
            for jj in range(self.SIZE-1):
                try:
                    cv2.imwrite('new'+str(ii)+'_'+str(jj)+'.png',recognizes_digits[ii][jj])
                except cv2.error:
                    pass
