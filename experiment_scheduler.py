import os
from datetime import datetime
import numpy as np

num_vertices = 64
#sparsities = [0.0625, 0.125, 0.25, 0.375, 0.5]
sparsities = [0.375, 0.5]

dataset_configs = [[102400,20480], [51200, 10240], [25600,5120]]


clone_fpath = "main_"+str(np.random.randint(30000))+".py"
os.system("cp main.py %s"%(clone_fpath))

for sparsity in sparsities:
	for dataset_config in dataset_configs:
		tag = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
		log_fp = "%d_%.2f_%d_%d_%s.txt"%(num_vertices, 100*sparsity, dataset_config[0], dataset_config[1], tag)

		cmd = "python %s --num-vertices=%d  --sparsity=%f --train-samples=%d --test-samples=%d | tee %s"%(clone_fpath, num_vertices, sparsity, dataset_config[0], dataset_config[1], log_fp)
		#print(cmd)
		os.system(cmd)

		
		#key = "%d_%.2f_%d_%d"%(num_vertices, 100*sparsity, dataset_config[0], dataset_config[1])
		#os.system("grep L1 %s*.txt | tail -1"%(key))


os.system("rm %s"%(clone_fpath))

	
