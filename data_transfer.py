import boto3
import math
import os
import sys
import threading
import time
import urllib2

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.counter = 0

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        self.counter += 1
        with self._lock:
            self._seen_so_far += bytes_amount
            if self.counter % 20 == 0:
                percentage = (self._seen_so_far / self._size) * 100
                print("\r%s  %s / %s  (%.2f%%)" % (self._filename, self._seen_so_far, self._size,percentage))


def upload_file(bucket_name, filename, s3_path):
    session = boto3.Session()
    s3_client = session.client( 's3' )

    try:
        print("Uploading file: {}".format(filename))

        tc = boto3.s3.transfer.TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            max_concurrency=10,
            num_download_attempts=10,
            multipart_chunksize=16 * 1024 * 1024,
            max_io_queue=10000
        )
        t = boto3.s3.transfer.S3Transfer(client=s3_client, config=tc )

        t.upload_file(filename, bucket_name, s3_path, callback=ProgressPercentage(file_name))

    except Exception as e:
        print("Error uploading: {}".format(str(e)))


url = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_0.tar"
file_name = url.split('/')[-1]
u = urllib2.urlopen(url)
with open(file_name, 'w') as fname:
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 5000000
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        fname.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status)

start_time = time.time()
upload_file("emrbucket-dag20180305", file_name, file_name)
upload_time = round((time.time() - start_time) / 60, 2)
print("File {} upload took {} minutes".format(file_name, upload_time))
print("Removing file {}".format(file_name))
os.remove(file_name)