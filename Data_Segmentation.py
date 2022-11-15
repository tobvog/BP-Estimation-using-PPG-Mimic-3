## \brief
#

import numpy as np
from Datasorting_mimic3 import Datasorting_mimic3

#%%
### Security Comment ###
path = "D:/MIMICIII_Database/mimic3wdb-1.0.physionet.org/"
#path_own = "D:/MIMICIII_Database/segmented_data_slapnicar/"
path_own = "D:/MIMICIII_Database/Test/"
### Security Comment ###

sorting_ = Datasorting_mimic3()
            
def first_loop(temp,i,j,add,path_o):
    data_test = []
    for k in range(1,10):
        add2 = "_000"
        path_name = str(temp)+str(add)+str(add2)+str(k)
        if sorting_.check_existence(path_name+".hea") == True:
            if sorting_.check_data_size(path_name+".dat") == True and sorting_.check_pleth_abp(path_name) == True:
                if sorting_.check_time_length(path_name) >= 10:
                    print(path_name+" passed")
                    temp = sorting_.extract_data(path_name)
                    data_test.append(temp)
                    np.save(path_o+str(add)+str(add2)+str(k), temp)
                    #np.savez_compressed(path_o+str(add)+str(add2)+str(k), temp)
        else:
            print("break")
            break
    
    for k in range(10,100):
        add2 = "_00"
        path_name = str(temp)+str(add)+str(add2)+str(k)
        print(path_name)
        if sorting_.check_existence(path_name+".hea") == True:
            if sorting_.check_data_size(path_name+".dat") == True and sorting_.check_pleth_abp(path_name) == True:
                if sorting_.check_time_length(path_name) >= 10:
                    print(path_name+" passed")
                    temp = sorting_.extract_data(path_name)
                    data_test.append(temp)
                    np.save(path_o+str(add)+str(add2)+str(k), temp)                                        
        else:
            print("break")
            break
                
    for k in range(100,1000):
        add2 = "_0"
        path_name = str(temp)+str(add)+str(add2)+str(k)
        if sorting_.check_existence(path_name+".hea") == True:
            if sorting_.check_data_size(path_name+".dat") == True and sorting_.check_pleth_abp(path_name) == True:  
                if sorting_.check_time_length(path_name) >= 10:
                    print(path_name)
                    temp = sorting_.extract_data(path_name)
                    data_test.append(temp)
                    np.save(path_o+str(add)+str(add2)+str(k), temp)
        else:
            print("break")
            break  
               
    for k in range(1000,10000):
        add2 = "_"
        path_name = str(temp)+str(add)+str(add2)+str(k)
        if sorting_.check_existence(path_name+".hea") == True:
            if sorting_.check_data_size(path_name+".dat") == True and sorting_.check_pleth_abp(path_name) == True: 
                if sorting_.check_time_length(path_name) >= 10:
                    print(path_name+" passed")
                    temp = sorting_.extract_data(path_name)
                    data_test.append(temp)
                    np.save(path_o+str(add)+str(add2)+str(k), temp)
        else:
            print("break")
            break

    #return data


###############################################################
#################### Main #####################################
###############################################################

sorted_data = []

for i in range(30,40):

    for j in range(1,10):
        add = str(i)+"0000"+str(j)
        path_name = str(i)+"/"+str(i)+"/"+add+"/"
        if sorting_.check_existence(path_name) == True:
            first_loop(path_name,i,j,add,path_own)
                        
    for j in range(10,100):
        add = str(i)+"000"+str(j)
        path_name = str(i)+"/"+str(i)+"/"+add+"/"
        if sorting_.check_existence(path_name) == True:
            first_loop(path_name,i,j,add,path_own)
             
    for j in range(100,1000):
        add = str(i)+"00"+str(j)
        path_name = str(i)+"/"+str(i)+"/"+add+"/"
        if sorting_.check_existence(path_name) == True:
            first_loop(path_name,i,j,add,path_own)  
            
    for j in range(1000,10000):
        add = str(i)+"0"+str(j)
        path_name = str(i)+"/"+str(i)+"/"+add+"/"
        if sorting_.check_existence(path_name) == True:
            first_loop(path_name,i,j,add,path_own)     
            
    for j in range(10000,100000):
        add = str(i)+str(j)
        path_name = str(i)+"/"+str(i)+"/"+add+"/"
        if sorting_.check_existence(path_name) == True:
            first_loop(path_name,i,j,add,path_own)
                        
    if i == 30:
        path_name_old = str(i)+"/"+str(i)+"/"+add+"/"
        path_name_new = str(i)+"/"+add+"/"
        
        for j in range(1,10):
            add = str(i)+"0000"+str(j)
            path_name_old = str(i)+"/"+str(i)+"/"+add+"/"
            path_name_new = str(i)+"/"+add+"/"
            if sorting_.check_existence(path_name_new) == True and sorting_.check_existence(path_name_old) == False:
                first_loop(path_name_new,i,j,add,path_own)  
                           
        for j in range(10,100):
            add = str(i)+"000"+str(j)
            path_name_old = str(i)+"/"+str(i)+"/"+add+"/"
            path_name_new = str(i)+"/"+add+"/"
            if sorting_.check_existence(path_name_new) == True and sorting_.check_existence(path_name_old) == False:
                first_loop(path_name_new,i,j,add,path_own)   
                
        for j in range(100,1000):
            add = str(i)+"00"+str(j)
            path_name_old = str(i)+"/"+str(i)+"/"+add+"/"
            path_name_new = str(i)+"/"+add+"/"
            if sorting_.check_existence(path_name_new) == True and sorting_.check_existence(path_name_old) == False:
                first_loop(path_name_new,i,j,add,path_own)  
                
        for j in range(1000,10000):
            add = str(i)+"0"+str(j)
            path_name_old = str(i)+"/"+str(i)+"/"+add+"/"
            path_name_new = str(i)+"/"+add+"/"
            if sorting_.check_existence(path_name_new) == True and sorting_.check_existence(path_name_old) == False:
                first_loop(path_name_new,i,j,add,path_own)    
                
        for j in range(10000,100000):
            add = str(i)+str(j)
            path_name_old = str(i)+"/"+str(i)+"/"+add+"/"
            path_name_new = str(i)+"/"+add+"/"
            if sorting_.check_existence(path_name_new) == True and sorting_.check_existence(path_name_old) == False:
                first_loop(path_name_new,i,j,add,path_own)
 
#%%





