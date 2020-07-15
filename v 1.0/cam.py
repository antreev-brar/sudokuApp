import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28,1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = larger_model()
model.summary()
dataset_labels =np.array(['1', '2', '3' ,'4', '5', '6', '7' ,'8', '9', '~.'])

# load model
model = load_model('./model.h5')
# summarize model.
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def cross (a,b):
    return [str1 + str2 for str1 in a for str2 in b ]
rows='ABCDEFGHI'
columns = '123456789'
def make_puzzle_grid():
    return (cross(rows,columns))
cells = make_puzzle_grid()
def sudoku_solver(map_input_grid):
  final = {}
  #map_input_grid_={'A1':8,'B3':3,'B4':6,'C2':7,'C5':9,'C7':2,'D2':5,'D6':7,'E5':4,'E6':5,'E7':7,'F4':1,'F8':3,'G3':1,'G8':6,'G9':8,'H3':8,'H4':5,'H8':1,'I2':9,'I7':4}
  map_input_grid_= map_input_grid.copy()
  #map_input_grid_= {'A4': 6, 'A6': 4, 'A7': 7, 'B1': 7, 'B3': 6, 'B9': 9, 'C6': 5, 'C8': 8, 'D2': 7, 'D5': 2, 'D8': 9, 'D9': 3, 'E1': 8, 'E9': 5, 'F1': 4, 'F2': 3, 'F5': 1, 'F8': 7, 'G2': 5, 'G4': 2, 'H1': 3, 'H7': 2, 'H9': 8, 'I3': 2, 'I4': 3, 'I6': 1}
  #map_input_grid_={'A1': 5, 'A2': 3, 'A5': 7, 'B1': 6, 'B4': 1, 'B5': 9, 'B6': 5, 'C2': 9, 'C3': 8, 'C8': 6, 'D1': 8, 'D5': 6, 'D9': 3, 'E1': 4, 'E4': 8, 'E6': 3, 'E9': 1, 'F1': 7, 'F5': 2, 'F9': 6, 'G2': 6, 'G7': 2, 'G8': 8, 'H4': 4, 'H5': 1, 'H6': 9, 'H9': 5, 'I5': 8, 'I8': 7, 'I9': 9}
  #print(map_input_grid_)
  def cross (a,b):
      return [str1 + str2 for str1 in a for str2 in b ]
  rows='ABCDEFGHI'
  columns = '123456789'
  def make_puzzle_grid():
      return (cross(rows,columns))
  cells = make_puzzle_grid()


  grid ={}
  map_input_grid ={}
  for i in cells:
      grid[i]=[1,1,1,1,1,1,1,1,1] 
  #print(grid['I8'])


  num_possibilities={}
  done = 0
  for x in cells :
   num_possibilities[x] = grid[str(x)].count(1) 
  #print (num_possibilities)


  def make_input(dict):
      for i in dict:
          grid[i]=[1 if x == dict[i] else 0 for x in range(1,10)]
          num_possibilities[i]=1
      #print(grid)
      #print(num_possibilities)
  make_input(map_input_grid_)
  #make_input(map_input_grid)


  def filter(cell , val, grid__ ,num_possibilities__):
    for i in cross(cell[0],columns):
        if i == cell : 
            continue
        if grid__[i][val-1]==1:
            num_possibilities__[i] -= 1
            grid__[i][val-1]= 0
    for i in cross(rows,cell[1]):
        if i == cell : 
            continue
        if grid__[i][val-1] == 1 :
            num_possibilities__[i] -= 1
            grid__[i][val-1] = 0
    if cell[0]=='A' or cell[0]=='B' or cell[0]=='C':
            row_ = 'ABC'
    if cell[0]=='D' or cell[0]=='E' or cell[0]=='F':
            row_ = 'DEF'
    if cell[0]=='G' or cell[0]=='H' or cell[0]=='I':
            row_ = 'GHI'
    if cell[1]=='1' or cell[1]=='2' or cell[1]=='3':
            _column = '123'
    if cell[1]=='4' or cell[1]=='5' or cell[1]=='6':
            _column = '456'
    if cell[1]=='7' or cell[1]=='8' or cell[1]=='9':
            _column = '789'
    for i in cross(row_,_column):
        if i == cell : 
            continue
        if grid__[i][val-1] == 1 :
            num_possibilities__[i] -= 1
            grid__[i][val-1] = 0
    return  dict(num_possibilities__),dict(grid__) 
  #grd , num_posbilities_ = filter('A1',7,grid ,num_possibilities)
  #print(num_possibilities)
  #print(grid)


  def print_poss(num_poss_):
    for i in cells:
        if(num_poss_[i]=='.'):
            print(num_poss_[i], end ="  ")
        else :
            print(num_poss_[i], end ="  ")
        if i[1]=='9':
            print(" ")
    print("     ")


  def something(grid_,num_possibilities_,done,m,mc):
    mini = 10
    
    min_cell = 'x'
    while (True):
      mini = 10
      min_cell = 'x'
      flag = 1

    
    
      for i in cells:
          if (num_possibilities_[i] != '.' and num_possibilities_[i] < mini ):
            mini = num_possibilities_[i]
            min_cell = i
          if num_possibilities_[i]==1 :
            flag = 0
            #print(grid_)
            num_possibilities_,grid_   = filter(i, grid_[i].index(1)+1 ,grid_ , num_possibilities_)
            num_possibilities_[i]='.'
      #print(mini,min_cell)
      #if(m == 8 and mc =='B5'):
      #print_poss(num_possibilities_)
      for i in cells:
        if (num_possibilities_[i]!='.'):
            if (num_possibilities_[i]<1):
                #print('*********')
                #print(grid_)
                #print(num_possibilities_)
                #print('*********')
                return 0
      x=1
      for check in cells:
        if num_possibilities_[check] != '.':
            x=0
      if(x):
        for i in cells:
            print(grid_[i].index(1)+1, end ="     ")
            if i[1]=='9':
              print(" ")
            final[i]=grid_[i].index(1)+1
            #print(grid_)
        return 1


      if(flag and mini==10):
        done = 1
        if(m == 8 and mc =='B5'):
              print_poss(num_possibilities_)
              print(grid_)
              for ij in cells:
                    print(grid_[ij].index(1)+1, end ="     ")
                    if ij[1]=='9':
                        print(" ")
        return done
        break
      #print(mini)
      if mini != 1 and mini < 10:
            #print('breaking')
            break

    possibilities = [ x+1 for x in range(0,9)  if grid_[min_cell][x] == 1]
  
    for u in range(len(possibilities)) :
                j=possibilities[u]
                '''
                if(min_cell=='B5' and j == 8):
                  print('----------------------------------------------------------------------------------') 
                  print_poss(num_possibilities_)
                  print(grid_)
                  print(mini ,min_cell , possibilities)
                  
                ''' 
                #print("now exploring ",j)
                #print('at ',min_cell) 
                #print('----------------------------')
                #print_poss(num_possibilities_)
                #print(grid_)
                #print('----------------------------')
                temp_num_possibilities = dict(num_possibilities_)
                temp_num_possibilities[min_cell] = 1
                
                temp = {}
                for k in cells:
                    temp[k] = list(grid_[k])
                temp[min_cell] = [0,0,0,0,0,0,0,0,0]
                temp[min_cell][j-1] = 1 
                #print('before----------------------------')
                #print(grid_)
                #print(num_possibilities_)
                done = something(temp,temp_num_possibilities,done,j,min_cell)
                if(done): return 1
                #print('after----------------------------')
                #print(grid_)
                #print(num_possibilities_)
                #if(u==len(possibilities)-1): 
                #  return 0
      #print('at the end of while')
  _grid={}
  for k in cells:
    _grid[k] = list(grid[k])
  _num_possibilities = num_possibilities
  #print_poss(num_possibilities)
  something(_grid,_num_possibilities,done,7,'7')
  return final

def plot_many_images(images,titles,rows=1 ,columns = 2):
  for i , image in enumerate(images):
    plt.subplot(rows,columns , i+1)
    plt.imshow(image,gray)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
  plt.show()

def pre_process_image(image, skip_dilate=False):
  proc = cv2.GaussianBlur(image.copy(),(9,9), 0)
  #proc = cv2.Canny(proc,100,150)
  proc = cv2.adaptiveThreshold(proc , 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 11 ,2)
  
  proc = cv2.bitwise_not(proc , proc)
  if not skip_dilate:
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)
  return proc

def find_corners_largest_polygon(img):
  #print('in corners')
  contours ,h = cv2.findContours(img.copy() ,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
  contours    =  sorted(contours, key = cv2.contourArea , reverse =True)
  polygon     =  contours[0]
  bottom_right,_ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  bottom_left,_ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  top_right,_ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  top_left,_ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
  return [polygon[top_left][0] , polygon[top_right][0] , polygon[bottom_right][0] , polygon[bottom_left][0] ]

def distance_btw(p1,p2):
  a= p1[0]-p2[0]
  b= p1[1]-p2[1]
  return np.sqrt((a ** 2) + (b ** 2))

def crop_and_wrap(img ,crop_rect):
  top_left,top_right,bottom_right , bottom_left = crop_rect[0],crop_rect[1],crop_rect[2],crop_rect[3]
  height = max(bottom_left[1] - top_left[1] ,bottom_right[1] - top_right[1])
  width = max(top_right[0] - top_left[0] , bottom_right[0] - bottom_left[0])
  print([height , width ])
  src = np.array([top_left,top_right,bottom_right , bottom_left],dtype='float32')
  side = max([
              distance_btw(top_left,top_right),
              distance_btw(bottom_left,bottom_right),
              distance_btw(top_left,bottom_left),
             distance_btw(top_right,bottom_right)
  ])
  dst = np.array([[0,0] , [ side-1 , 0] , [side-1 , side -1] , [0 ,side -1]],dtype='float32')
  m = cv2.getPerspectiveTransform(src, dst)
  transformation_data = {
        'matrix' : m,
        'original_shape': (height, width)
    }
  warped = cv2.warpPerspective(img, m, (int(side), int(side)))
  
 
  return warped , transformation_data

def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))

def infer_grid(img):
  squares = []
  side = img.shape[:1]
  side = side[0]/9

  for j in range (9):
    for i in range(9):
      p1 = (i * side ,j*side)
      p2 = ((i+1) * side ,(j+1)*side)
      squares.append((p1,p2))

  return squares

def get_digits(img , squares , size):
  digits = []
  img = pre_process_image(img.copy(), skip_dilate=True)
  for square in squares:
    digits.append(extract_digit(img,square ,size))
  return digits 

def cut_from_rect(img,rect):
  return img[int(rect[0][1]):int(rect[1][1]) , int(rect[0][0]):int(rect[1][0])]

def find_largest_feature( inp_img , scan_tl=None,scan_br=None):
  img = inp_img.copy()
  height,width =img.shape[:2]
  
  max_area =0
  seed_point =(None ,None)
  if scan_tl is None:
    scan_tl = [0, 0]

  if scan_br is None:
    scan_br = [width, height]

  for x in range(scan_tl[0], scan_br[0]):
    for y in range(scan_tl[1], scan_br[1]): 
      if img.item(y,x) == 255 and x < width  and y<height:
        area =cv2.floodFill(img ,None ,(x,y),64)
        if area[0] > max_area :
          max_area =area[0]
          seed_point=(x,y)
  for x in range(width):
     for y in range(height):
       if img.item(y,x) == 255 and x < width and y < height:
         cv2.floodFill(img , None ,(x,y) , 64)

  mask = np.zeros((height+2 ,width +2),np.uint8)

  if all([p is not None for p in seed_point]):
    cv2.floodFill(img , mask , seed_point ,255)
  
  top ,bottom ,left ,right = height ,0 , width ,0 

  for x in range(width):
    for y in range(height):
      if img.item(y,x)==64 :
        cv2.floodFill(img ,mask , (x,y) , 0)
      
      if img.item(y,x)==255:
        top = y if y < top else top
        bottom = y if  y > bottom else bottom
        left = x if x < left else left
        right = x if x > right else right

  bbox = [[left,top],[right , bottom]]
  return img , np.array(bbox , dtype='float32'),seed_point

def extract_digit(img,rect ,size):
  digit = cut_from_rect(img ,rect)
  h,w = digit.shape[:2]
  margin =int(np.mean([h,w])/2.5)
  
  _,bbox ,seed =find_largest_feature(digit , [margin ,margin],[w-margin , h-margin])
  digit = cut_from_rect(digit,bbox)
  w = bbox[1][0] - bbox[0][0]
  h = bbox[1][1] - bbox[0][1]
  if w>0 and h>0 and (w*h)>100 and len(digit)>0:
    return scale_and_centre(digit ,size ,4)
  else:
    return np.zeros((size,size),np.uint8)

def show_digits(digits , colour =255):
  rows =[]
  with_border = [cv2.copyMakeBorder(img.copy(),1,1,1,1,cv2.BORDER_CONSTANT ,None ,colour ) for img in digits]
  for i in range(9):
    row = np.concatenate(with_border[i*9 : ((i+1)*9)] , axis =1)
    rows.append(row)
  show_image(np.concatenate(rows))

def show_image(img):
  cv2.imshow('image_of_something',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def digit_grid(digits):
  plt.figure(figsize=(10,9))
  plt.subplots_adjust(hspace=0.5)
  predict_digits = [x.astype('float32') / 255 for x in digits]
 
  tf_model_predictions = model.predict(np.expand_dims(predict_digits,axis=-1))
  #print(tf_model_predictions)
  predicted_ids = np.argmax(tf_model_predictions, axis=-1)
  #print(np.min(np.max(tf_model_predictions, axis=-1)))
  min_confidence = np.min(np.max(tf_model_predictions, axis=-1))
  if(min_confidence < 0.998):
      return 0
  predicted_labels = dataset_labels[predicted_ids]  
  map_input = make_input_map(predicted_labels)
  '''
  plt.figure(figsize=(10,9))
  plt.subplots_adjust(hspace=0.5)
  for n in range(81):
    plt.subplot(9,9,n+1)
    plt.imshow(digits[n].reshape(28,28))
    color = "green"
    plt.title(predicted_labels[n].title(), color=color)
    plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
   '''
  return map_input

def make_input_map(labels):

  temp = {}
  for i in range(81):
    if (labels[i]=='~.'): continue
    temp[cells[i]] = ord(labels[i]) - 48

  #print(temp)
  return temp

def displaySolution(image,map_, final , frame , coords ):
    image = image.copy()
    tl, tr, br, bl = coords

    cell_width = image.shape[1] // 9
    cell_height = image.shape[0] // 9

    for i in range(81):
        if cells[i] in map_:
            color = (0, 0, 0)
        else:
            color = (255, 0, 0)
            if(cells[i] not in final):
                return np.array([0])
			
            text = str(final[cells[i]])
            offsetX = cell_width // 15
            offsetY = cell_height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseline = cv2.getTextSize(text, font, fontScale = 1, thickness = 3)
            marginX = cell_width // 7
            marginY = cell_height // 7
            bottomLeft = tl[0]+ cell_width*(i%9) + (cell_width - text_width) // 2 + offsetX
            bottomRight = tl[1]+ cell_height*((i//9)+1) - (cell_height - text_height) // 2 + offsetY
            image = cv2.putText(frame, text, (int(bottomLeft), int(bottomRight)), font, 1, color, thickness = 3, lineType = cv2.LINE_AA)
    return image

def unwrap(image,original, coords):
    ratio = 1.0
    tl, tr, br, bl = coords
    print(original.shape)
    heightreal, widthreal = original.shape
    widthA = np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[1])**2)
    widthB = np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[1])**2)
    heightA = np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[1])**2)
    heightB = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[1])**2)
    width = max(widthA, widthB) * ratio
    height = width

    destination = np.array([
    [0, 0],
    [height, 0],
    [height, width],
    [0, width]], dtype = np.float32)
    M = cv2.getPerspectiveTransform( np.float32(coords),destination )
    unwarped = cv2.warpPerspective(image, M, (int(height), int(width)),original, flags = cv2.WARP_INVERSE_MAP)
    return unwarped
def perspective_transform(img, transformation_matrix, original_shape=None):
    warped = img

    if original_shape is not None:
        if original_shape[0]>0 and original_shape[1]>0:
            warped = cv2.resize(warped, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

    white_image = np.zeros((640, 480, 3), np.uint8)

    white_image[:,:,:] = 255

    # warped = cv2.warpPerspective(warped, transformation_matrix, (640, 480), borderMode=cv2.BORDER_TRANSPARENT)
    warped = cv2.warpPerspective(warped, transformation_matrix, (640, 480))

    return warped


def blend_non_transparent(face_img, overlay_img):
    # Let's find a mask covering all the non-black (foreground) pixels
    # NB: We need to do this on grayscale version of the image
    gray_overlay = overlay_img
    #cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]

    # Let's shrink and blur it a little to make the transitions smoother...
    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))

    # And the inverse mask, that covers all the black (background) pixels
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def parse_grid(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  original = gray
  #cv2.imread(path , cv2.IMREAD_GRAYSCALE)
  #cv2.imshow('gray_original',original)
  processed = pre_process_image(original)
  cv2.imshow('processed',processed)
  corners = find_corners_largest_polygon(processed)
  area = cv2.contourArea(np.float32(corners))
  print(area)
  
  if(area < 160000):
      cv2.imshow('cropped but solved',frame)
      return
  cropped , transformation = crop_and_wrap(original,corners)
  transformation_matrix = transformation['matrix']
  original_shape = transformation['original_shape']
  transformation_matrix = np.linalg.pinv(transformation_matrix)
  #cv2.imshow('cropped',cropped)
  squares = infer_grid(cropped)
  digits = get_digits(cropped , squares , 28)
  #show_digits(digits)
  map_input_grid = digit_grid(digits)
  if(map_input_grid == 0):
      cv2.imshow('cropped but solved',frame)
      return
  #print(map_input_grid)
  #make_input(map_input)
  #print(map_input_grid)
  final = sudoku_solver(map_input_grid)
  #print(final)
  solvedImage = displaySolution(cropped , map_input_grid, final, frame ,corners)
  if(solvedImage.all == 0):
      cv2.imshow('cropped but solved',frame)
      return
  #img_sudoku_final = perspective_transform(solvedImage, transformation_matrix, original_shape)
  #cv2.imshow('cropped but solved',img_sudoku_final)
  #img_final = blend_non_transparent(frame, img_sudoku_final)
  #result = unwrap(solvedImage.copy(),gray.copy() , corners)
  #print(img_final.shape)
  #cv2.imshow('result',img_final)
  cv2.imshow('cropped but solved',solvedImage)

def main():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        
        parse_grid(frame)
        # Display the resulting frame
        #cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
  main()


