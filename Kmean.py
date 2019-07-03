import csv
import numpy as np
import sys

ID_team = sys.argv[0] #file name is ID team
Input = sys.argv[1] #input type csv
Output_model = sys.argv[2] #output model
Output_asgn = sys.argv[3] # output assigments
K = int(sys.argv[4]) #K clusters in Kmean

INT_MAX = sys.maxsize   #define int_max

def init_K(k,data,numTitle):
    # element in list_K is the mean of data * 1/k
    list_K = np.array([]).reshape(0,numTitle)
    div,mod = divmod(len(data),k)
    index_temp = []
    for i in range(1,k): #make index to for loop in data
        index_temp.append(div*i)
    index_temp.append(len(data))
    
    a=0
    for idx in index_temp:
        mean_idx = [np.mean(data[a:idx],axis=0)]
        mean_idx = np.asarray(mean_idx)
        list_K= np.concatenate((list_K,mean_idx)) #add point
    return list_K
def find_exist_point(data):
    #filter data and appearment occurrent
    exist_point,occurrent = np.unique(data,return_counts=True,axis=0) #remove the dulicate point and save times occurent
    return exist_point,occurrent
def compute_distance(point1,point2):
    #compute distance (x1,y1) and (x2,y2) by square 
    distance = np.sum((point1-point2)**2)
    return round(distance,4)
def compute_SSE(k,list_K,depend_point,exist_point,occurrent):
    #compute sum of squared errors
    SSE = 0
    for i in range(len(depend_point)):
        SSE += np.linalg.norm(list_K[depend_point[i]]-exist_point[i])*occurrent[i]
    return SSE

def Kmean(title,data,k):
    numTitle = len(title) #number of title
    numInstance = len(data) #number of instances 

    print("finding exist_point ...")
    exist_point,occurrent = find_exist_point(data) # exist_point is array which contain(i,j) where data[i][j]==1
    print("init list K ...")
    list_K = init_K(k,data,numTitle)     #initialize list k point follow mean
    
    #only browse the exist_point
    #run loop

    depend_point = [-1]*len(exist_point) #points depend on each k, it has number of row = exist_point's row
    iteration = 0
    print("Processing Kmean:")
    while(1): #clustering
        iteration+=1
        row_exist_point = 0  #index of depend_point
        temp_list_K = np.zeros((k,numTitle))
        count_list_K = [0]*k #to compute mean
        out = True # if dont update depend_point then out
        for point in exist_point:  #specify points depend on which K
            min_distance = INT_MAX
            index_temp = -1
            for i in list_K:  #compute distance and compare min/ update index has min distance
                index_temp+=1
                tmp_distance = compute_distance(point,i) #compute distance (x1,y1) and (x2,y2) by square error
                if (tmp_distance<min_distance):
                    min_distance = tmp_distance #update
                    idx = index_temp #index which min distance
            #end this loop: index which min distance
            if(depend_point[row_exist_point]!=idx): #compare and update
                depend_point[row_exist_point]=idx
                out = False
            row_exist_point+=1
        if (out == True): #out if not update really
            print("\nDone!")
            break
        #update list_K by compute mean all depended point
        for i in range(len(depend_point)):
            temp_list_K[depend_point[i]]+= exist_point[i]*occurrent[i]
            count_list_K[depend_point[i]]+=1*occurrent[i]
        for i in range(k):
            if (count_list_K[i] != 0):
                list_K[i]=temp_list_K[i]/count_list_K[i]

        #compute error
        sse = compute_SSE(k,list_K,depend_point,exist_point,occurrent)
        print( "iteration ",iteration,"...")
        print("SSE after ",iteration,"iteration(s): \t",sse )
     
    #save cluster data
    cluster_data = {} #cluster_data[i]:  contain element point depend on cluster i
    for i in range(K):
        cluster_data[i] = [] #init {i}:[]
    for i in range(len(depend_point)):
        for j in range(occurrent[i]): #add depend point to last column and append to cluster data
            cluster_data[depend_point[i]].append(np.hstack((exist_point[i],depend_point[i]))) 

    return cluster_data,list_K,depend_point,exist_point,occurrent
def read(fi):
    with open(fi) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        A = []
        for row in csv_reader:
            A.append(', '.join(row))
        
        title = A[0] #attribute
        title = title.split(", ")#split ","

        data = A[1:] #data
        data = [row.split(', ') for row in data]  #split ","
        data = [list( map(int,i) ) for i in data] #conver str to int
        data = np.asarray(data)

        numData = len(data) #number of data set
        numTitle = len(title) #number of title
    return data,title,numData,numTitle
def write_asgn(title,cluster_data,data,fo = "assignments.csv"):
    isTrue = False
    with open(fo, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE,  lineterminator='\n')
        title.append("Cluster")
        csv_writer.writerow(title)
        for i in data:
            isTrue = False
            for j in cluster_data:
                for k in cluster_data[j]:
                    if (np.array_equal(i,k[0:len(k)-1])):
                        csv_writer.writerow(k)
                        isTrue = True
                        break
                if (isTrue == True):
                    break
def write_model(k,title,numTitle,list_K,depend_point,exist_point,occurrent,cluster_data,fo="model.txt"):
    SSE = compute_SSE(k,list_K,depend_point,exist_point,occurrent)            #compute sum of squared error
    with open(fo,mode='w') as txt_file:
        #txt_file.write("k = %d\n" %k)
        txt_file.write("Within cluster sum of squared errors: %f\n" %SSE)
        txt_file.write("Cluster centroids:")
        txt_file.write("\t\t\t\t Cluster# \n")
        txt_file.write("Attribute \t")
        for i in range(k):
            txt_file.write("%d\t\t" %i)
        txt_file.write('\n')
        txt_file.write("\t\t")
        for i in range(k):
            txt_file.write("(%d)\t\t" %len(cluster_data[i]))
        txt_file.write('\n')
        txt_file.write("====================================================================================================================\n")
        for i in range(numTitle):
            if(i == 1 or i == (numTitle - 1)): #for "Products" and "Purchase" because the word's length is over one tab
                txt_file.write("%s\t"%title[i])
            else:
                txt_file.write("%s\t\t"%title[i])
            
            for j in range(k):
                txt_file.write("%f\t" %round(list_K[j][i],4))

            txt_file.write("\n")


def main():
    #read file and save data, title
    data,title,numData,numTitle = read(Input)
    #Kmean process
    cluster_data,list_K,depend_point,exist_point,occurrent = Kmean(title,data,K)
    #write assignments
    write_asgn(title,cluster_data,data,Output_asgn)
    #write model
    write_model(K,title,numTitle,list_K,depend_point,exist_point,occurrent,cluster_data,Output_model)

main()
