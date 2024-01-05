import glob 
import time 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk

file_name = glob.glob('./Taillard-PFSP/ta*') 

for fn in sorted(file_name):
    print("File Name: ",fn)
    def read_pfsp_instance(fn):
        with open(fn, "r") as f:
            num_jobs, num_machines = map(int, f.readline().split()) 
            p = {}
            for j in range(num_jobs):
                for index, value in enumerate(f.readline().split()[1::2]): 
                    p[j,index] = int(value) 
            return num_jobs, num_machines, p
    num_jobs, num_machines, p = read_pfsp_instance(fn)
    print(" Jobs: ", num_jobs, "Machines: ", num_machines, '\n', "Processing Times: ", p, '\n')



#############################################
#               NEH Algorithm               #
#############################################
def processing_time(num_jobs, num_machines, p_time):
    """
        STEP-1: Calculate the total time of each Job, in all Machines
    """
    descending_job_order = []
    total_times = []
    for j in range(num_jobs):
        sum = 0
        total_times_row = []
        for m in range(num_machines):
            sum = sum + p_time[j,m]
            total_times_row.append(p_time[j,m])
        total_times.append(total_times_row)
        descending_job_order.append((sum,j,total_times[j]))

    """
        STEP-2: Order the list in descending order
    """
    descending_job_order.sort(reverse=True)
    return descending_job_order

def order(num_jobs, num_machines, p_time): 
    descending_job_order = processing_time(num_jobs, num_machines, p_time) 
    job_order = []
    for j in range(num_jobs):
        # Insert in list -job_order- only the jobs
        job_order.append(descending_job_order[j][1])
    return job_order

def insertion(sequence, index_position, value): 
    new_seq = sequence[:]
    # Insert the value in the position that indicates the index_position variable for permutation
    new_seq.insert(index_position, value) 
    return new_seq

def makespan(new_seq, num_machines, p_time):
    # Creation of an array -job_makespan- that contains the execution times of each task on each machine
    job_makespan = np.zeros((len(new_seq),num_machines))
    for i, v in enumerate(new_seq):  
        for m in range(num_machines): 
            if i == 0 and m == 0:
                job_makespan[i,m] = p_time[v,m]  
            elif i == 0:
                job_makespan[i,m] = job_makespan[i,m-1] + p_time[v,m]
            elif m == 0:
                job_makespan[i,m] = job_makespan[i-1,m] + p_time[v,m]
            else:
                job_makespan[i,m] = max(job_makespan[i,m-1],job_makespan[i-1,m]) + p_time[v,m]
    return job_makespan

def neh_algorithm(num_jobs, num_machines, p_time):
    """
        STEP-3 & STEP-4: Take the Jobs from the descending list and calculate the makespan for each possible sequences
    """
    job_order = order(num_jobs, num_machines, p_time)
    sequence = [job_order[0]]
    for i in range(1,num_jobs):
        min_cmax = float("inf")
        for j in range(0, i + 1): 
            temp_seq = insertion(sequence, j, job_order[i]) 
            currmax_tmp =  makespan(temp_seq, num_machines, p_time)[len(sequence),num_machines-1] # the value of execution of last job in the last machine
            if min_cmax > currmax_tmp:
                min_cmax = currmax_tmp
                best_seq = temp_seq
        sequence = best_seq
    # Returns the best Sequence, the array that contains the execution times of each task on each machine, and the total makespan
    return sequence, makespan(sequence, num_machines, p_time), min_cmax



########################################
#               Graphics               #
########################################
def display_makespan(sequence,p_time):
    # Set as -job_order- the best sequence
    job_order = sequence[0]

    njobs, nmachines = sequence[1].shape

    job_makespan = np.zeros((njobs,nmachines))
    
    for j in range(njobs):
        for m in range(nmachines):
            # Set as -job_makespan[j,m]- the value of execution times of each task on each machine
            job_makespan[j,m] = sequence[1][j][m]

    colors = list(mcolors.TABLEAU_COLORS.values())
    plt.figure(figsize=(15, nmachines*1.5))
    # Sort the array with argsort(), so the parameter -left- in plt.barth can't take negative values
    sort = np.argsort(job_order) # πχ [4,3,2,0,1] => sort=[(3),(4),(2),(1),(0)] = [0,1,2,3,4]
    job_makespan2 = job_makespan[sort]
    for m in range(nmachines):
        for j in job_order:
            plt.barh(m, width=p_time[j,m], left=job_makespan2[j,m] - p_time[j,m], height=0.8, color=colors[j % len(colors)], label=f"Job {j}" if m==0 else "")
   
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.yticks(range(nmachines), [f"Machine {m}" for m in range(nmachines)])
    plt.xlabel('Time')
    plt.title('Permutation Flowshop Schedule')
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")   
    plt.gca().invert_yaxis() 
    plt.show()    



######################################
#               Window               #
######################################
def show(): 
    global text
    text =choosen_value.get()
    root.quit()

# Create the main window
root = tk.Tk()
root.geometry("500x350")
root.title("Permutation Flow-shop Scheduling Problem")
root.resizable(width=False, height=False)
root['padx']=0
root['pady']=100

# Create Label 
label = tk.Label( root , wraplength=400, text = "Select the file you want to run: " ) 
label.config(
    font=("Arial italic",14),
)
label.pack()

# Create a variable to store the selected option
choosen_value = tk.StringVar(root)

# Define options for dropdown menu
file_name = glob.glob('./Taillard-PFSP/ta*') 
options = []
for fn in sorted(file_name):
    options.append(fn)

# Create the drop down menu
dropdown_menu = tk.OptionMenu(root, choosen_value, *options)
dropdown_menu.pack(pady=20)
dropdown_menu.config(
    bg = 'white',
    fg = "black",
    font=("Arial",12),
    border=2
)

# Set the default selected option
choosen_value.set(options[0])

# Create button
button = tk.Button(
    root , text = "confirm choice" , 
    width=15, 
    font=12, 
    bg="#0052cc", 
    fg='#ffffff', 
    activebackground='#0052cc', 
    activeforeground='#aaffaa',
    command = show ).pack() 

# Run the application
root.mainloop()



####################################
#               MAIN               #
####################################
fileName = text
print("fileName: ", fileName)
start_time = time.time()
n_jobs, n_machines, p = read_pfsp_instance(fileName)
total_completion_time = processing_time(n_jobs, n_machines, p)
neh = neh_algorithm(n_jobs, n_machines, p)
end_time = time.time()
print("neh_algorithm: ", neh)
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
display_makespan(neh, p)