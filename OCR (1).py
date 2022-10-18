import numpy as np
from PIL import Image


def read_image(path):
  return np.asarray(Image.open(path).convert('L'))

TEST_DATA_FILENAME ='data/t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME ='data/t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME ='data/train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME ='data/train-labels.idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data,'big')
def read_images(filename,n_max_images=None):
    images=[]
    with open(filename,'rb') as f:
        _=f.read(4)
        n_images=bytes_to_int(f.read(4))
        if n_max_images:
            n_images=n_max_images
        n_rows=bytes_to_int(f.read(4))
        n_columns=bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image=[]
            for row_idx in range(n_rows):
                row=[]
                for columns_idx in range(n_columns):
                    pixel=f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
            
    return images  
def read_labels(filename,n_max_labels=None):
     labels=[]
     with open(filename,'rb') as f:
         _=f.read(4)
         n_labels=bytes_to_int(f.read(4))
         if n_max_labels:
             n_labels=n_max_labels
         for label_idx in range(n_labels):
             label=bytes_to_int(f.read(1))             
             labels.append(label)       
     return labels
def flatten_list(l):
    return[pixel for sublist in l for pixel in sublist]
    
def extract_feat(X):
    return [flatten_list(sample) for sample in X]
def dist(x,y):
    return sum([(bytes_to_int(x_i) - bytes_to_int(y_i))**2 for x_i,y_i in zip(x,y)])**(0.5)
    
def get_euclid_dist_for_test_sample(X_train,test_sample):
    return[dist(train_sample,test_sample) for train_sample in X_train]

def most_frequent(l):
    return max(l, key=l.count)

def knn(X_train,Y_train,X_test,k):
    y_pred=[]
    for test_sample_idx,test_sample in enumerate(X_test):
        training_distances=get_euclid_dist_for_test_sample(X_train,test_sample)
        sorted_indecies=[
            pair[0]
            for pair in  sorted(
                 enumerate(training_distances),
                 key=lambda x: x[1])
            ]
        candidates =[
            Y_train[idx]
            for idx in sorted_indecies[:k]
        ]
        top_candidate =most_frequent(candidates)
        y_pred.append(top_candidate)
    return y_pred   
    
     
 
    
def main():
   X_train =read_images(TRAIN_DATA_FILENAME,1000)
   Y_train=read_labels(TRAIN_LABELS_FILENAME,1000)
   X_test =read_images(TEST_DATA_FILENAME,5)
   Y_test=read_labels(TEST_LABELS_FILENAME,5)
   
   #for idx,test_sample in enumerate(X_test):
    #write_image(test_sample,f'{TEST_DIR}{idx}.png')
   
   X_test=[read_image(f'data/our_test.png')] 
   Y_test = [2]
   X_train = extract_feat(X_train)
   X_test = extract_feat(X_test)
 
   y_pred=knn(X_train,Y_train,X_test,7)
 
   accuracy= sum([
        int(y_pred_i == y_test_i)
        for y_pred_i,y_test_i
        in zip(y_pred,Y_test)
        ])/len(Y_test)
   
   print(y_pred)
   print(f"accuracy is {accuracy*100}%")
if __name__=='__main__':
    main()


 