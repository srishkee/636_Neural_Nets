#!/bin/bash 
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Cooking       #Set the job name to "JobExample2"
#SBATCH --time=00:02:30               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=28          #Request all 28 cores per node
#SBATCH --mem=90G                     #Request 8GB per node 
#SBATCH --output=output/cooking_log_%j.txt      #Send stdout/err to "Example2Out.[jobID]" 
#SBATCH --gres=gpu:2			# Per node
#SBATCH --partition=gpu

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122749846190	       #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=skumar55@tamu.edu      #Send all emails to email_address 

#User's Executable Line
module load Anaconda/3-5.0.0.1
source activate kumarenv
python predict.py