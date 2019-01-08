The results of the codes can be found in the corresponding pfd files. The report is in a seperate pdf file (report.pdf).

I wrote my codes in MATLAB. To run the codes, first put the .m files hw2_q3a.m, hw2_q3b.m and hw2_q4.m to the same folder. They are currently in the Codes folder.

Open matlab, open the folder in which the above files are. Clear all variables. (clear variables command). Then run hw2_q3a FIRST in MATLAB in the following manner: hw2_q3a('path')
where path defines the path to the folder which contains HW2data.mat file. The path shouldn't contain the filename, the code does this automatically.

For example in my computer I use: hw2_q3a('C:\Users\User\Desktop\B8\CS 464\Homeworks\HW2\') to run the first part.

The program will run and give the answers as relevant plots and information in the command line.
It will also save the test and train sets and their labels as .mat files to the current directory.

After running hw2_q3a, hw2_q3b and hw2_q4 can be run without parameters such as: hw2_q3b() and hw2_q4(), because
the train and test sets were saved to the current directory by the hw2_q3a.m code. 

Also, hw2_q3a and hw2_q3b codes save the desired prediction results to the current folder with the names:
linear_SVM_Results.csv and gaussian_SVM_Results.csv respectively. 


