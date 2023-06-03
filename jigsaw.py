import numpy as np
import random
import time

def main():
    field = np.zeros((4, 6), dtype=np.uint32)

    #while puzzle unsolved
    i = 0
    while True:
        i+=1
        form_number = random.randint(1,6)
        form = None
        # 1:green, 2:yellow, 3:brightblue, 4:darkblue, 5:red, 6:orange
        if(form_number == 1):
            form = np.array([[1, 0],[1, 1]], dtype=np.uint32)
        elif(form_number == 2):
            form = np.array([[1, 1],[0, 1]], dtype=np.uint32)
        elif(form_number == 3):
            form = np.array([[1, 1],[1, 1]], dtype=np.uint32)
        elif(form_number == 4):
            form = np.array([[1],[1],[1]], dtype=np.uint32)
        elif(form_number == 5):
            form = np.array([[1, 1, 0],[0, 1, 1]], dtype=np.uint32)
        elif(form_number == 6):
            form = np.array([[1]], dtype=np.uint32)
        
        print("round"+ str(i))
        print("field before")
        print(field)
        print("new form + row + col")
        print(form)
        print(form.shape[0]) # row
        print(form.shape[1]) # col

        # try to add somewhere / from upper left to lower right
        

        print("field after")
        print(field)
        print()
        time.sleep(2)

main()