# DefinitelyWildcats
DS1004 - Big Data Project

Data for this project can be found here
https://s3.console.aws.amazon.com/s3/buckets/emrbucket-dag20180305/ds1004-project/?region=us-east-1&tab=overview

## Initial Setup
1. Create the EMR cluster with the following settings:
![EMR_setup](assets/misc/aws-cluster-start.png?raw=true "Title")
1. Go to the homepage for your cluster details
   1. Open the Summary Tab
   1. Click on security groups for master under section "Security and access"
   1. Click the row with group name "ElasticMapReduce-master"
   1. Click "Edit"
   1. Scroll down to the bottom of the window that pops up and click "Add Rule"
   1. For the fields: select "SSH" for the type, leave the "port" set to 22 (this should be the default), and for the "source" select "Anywhere"
   1. Add a new TCP inbound rule for port 8888 (you will need this to access the jupyter server)
1. SSH into your EMR cluster
1. To install git run `sudo yum install git-core`
1. Add `export PYSPARK_PYTHON=/usr/bin/python3` to your bash profile (this will ensure pyspark runs with python3)
1. Add `export SPARK_HOME=/usr/lib/spark` (for some reason is not set by default - not sure how pyspark finds it???)

## Steps to get Jupyter Notebook running on EMR and then your local machine

1. Run `sudo python3 -m pip install --upgrade pip`
1. Run `sudo python3 -m pip install findspark`
1. Run `sudo python3 -m pip install numpy`
1. Run `sudo python3 -m pip install scipy`
1. Run `sudo python3 -m pip install sklearn`
1. Run `sudo python3 -m pip install awscli`
1. Run `sudo python3 -m pip install jupyter`
1. Run `jupyter notebook --no-browser --port=8888 --ip=0.0.0.0` (the notebook is now running at port 8888)
1. Note the token that is generated to enter into the browser at some later point in time.
1. Open a new terminal and input `ssh -i {}.pem -L {local port e.g. 8212}:localhost:8888 hadoop@{}.compute-1.amazonaws.com`
1. Go to http://localhost:8212/tree in Chrome
1. Voila!

Try running this
![EMR_sample](assets/misc/emr-jupyter-sample.png?raw=true "Title")


Note to self: some of these steps can be moved to bootstrap actions. Might not be worth it though since those need to be manually entered anyway at startup.

## Running the image processor
You can use spark submit on the image_analysis_centriods.py file specifying the data directory as a parameter at runtime.

