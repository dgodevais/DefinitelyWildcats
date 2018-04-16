# DefinitelyWildcats
DS1004 - Big Data Project

## Steps to get Jupyter Notebook running on EMR and then your local machine

1. Create the EMR cluster with the following settings:



1. Go to the homepage for your cluster details
   1. Open the Summary Tab
   1. Click on security groups for master under section "Security and access"
   1. Click the row with group name "ElasticMapReduce-master"
   1. Click "Edit"
   1. Scroll down to the bottom of the window that pops up and click "Add Rule"
   1. For the fields: select "SSH" for the type, leave the "port" set to 22 (this should be the default), and for the "source" select "Anywhere"
   1. Add a new TCP inbound rule for port 8888 (you will need this to access the jupyter server)
1. SSH into your EMR cluster
1. Run `jupyter notebook --no-browser --port=8888 --ip=0.0.0.0` (the notebook is now running at port 8888)
1. Open a new terminal and input `ssh -i {}.pem -L {local port e.g. 8212}:localhost:8888 hadoop@{}.compute-1.amazonaws.com`
