# ***************************************************************************************
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.                    *
#                                                                                       *
# Permission is hereby granted, free of charge, to any person obtaining a copy of this  *
# software and associated documentation files (the "Software"), to deal in the Software *
# without restriction, including without limitation the rights to use, copy, modify,    *
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to    *
# permit persons to whom the Software is furnished to do so.                            *
#                                                                                       *
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,   *
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A         *
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT    *
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION     *
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE        *
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                *
# ***************************************************************************************
import argparse
import datetime
import glob
import math
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import boto3


class S3Util:
    def upload_file(self, localpath, remote_path, quite_mode=False):
        """
    Uploads a file to s3
        :param quite_mode: If False, prints the status of each file downloaded
        :param localpath: The local path
        :param remote_path: The s3 path in format s3://mybucket/mydir/mysample.txt
        """

        start = datetime.datetime.now()

        bucket, key = self._get_bucketname_key(remote_path)

        if key.endswith("/"):
            key = "{}{}".format(key, os.path.basename(localpath))

        s3 = boto3.client('s3')

        s3.upload_file(localpath, bucket, key)

        if not quite_mode:
            download_time = datetime.datetime.now() - start

            print("Uploading file {} to {} in {} seconds".format(localpath, remote_path, download_time.total_seconds()))

    @staticmethod
    def _get_bucketname_key(uripath):
        assert uripath.startswith("s3://")

        path_without_scheme = uripath[5:]
        bucket_end_index = path_without_scheme.find("/")

        bucket_name = path_without_scheme
        key = "/"
        if bucket_end_index > -1:
            bucket_name = path_without_scheme[0:bucket_end_index]
            key = path_without_scheme[bucket_end_index + 1:]

        return bucket_name, key

    def download_file(self, remote_path, local_dir, quite_mode=False):
        """
        Download a single file from s3
        :param quite_mode: If False, prints the status of each file downloaded
        :param remote_path: The remote s3 file
        :param local_dir: The local directory to save the file to
        :return:
        """
        start = datetime.datetime.now()
        bucket, key = self._get_bucketname_key(remote_path)

        s3 = boto3.client('s3')

        local_file = os.path.join(local_dir, remote_path.split("/")[-1])

        # This is to avoid boto3 s3_download file attempting to create the same local path across multiple calls resulting in filnotfounderror
        if not os.path.exists(local_dir):
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        s3.download_file(bucket, key, local_file)

        if not quite_mode:
            download_time = datetime.datetime.now() - start

            print("Downloaded file from {} to {} in {} seconds".format(remote_path, local_file,
                                                                       download_time.total_seconds()))

    def download_object(self, remote_path, quite_mode=True):
        """
        Downloads binary bytes from s3 without saving file
        :param quite_mode: If False, prints the status of each file downloaded
        :param remote_path: The remote s3 path
        :return: returns binary bytes from s3 without saving file
        """
        start = datetime.datetime.now()

        bucket, key = self._get_bucketname_key(remote_path)

        s3 = boto3.client('s3')

        s3_response_object = s3.get_object(Bucket=bucket, Key=key)
        object_content = s3_response_object['Body'].read()

        if not quite_mode:
            download_time = datetime.datetime.now() - start

            print("Downloaded object {} in {} seconds ".format(remote_path, download_time.total_seconds()))

        return object_content

    def list_files(self, remote_path):
        """
Lists the files in s3
        :param remote_path: The s3 uri, e.g. s3://mybucket/prefix/
        :return: List of files
        """
        assert remote_path.startswith("s3://")
        assert remote_path.endswith("/")

        bucket, key = self._get_bucketname_key(remote_path)

        s3 = boto3.resource('s3')

        bucket = s3.Bucket(name=bucket)

        return ((o.bucket_name, o.key) for o in bucket.objects.filter(Prefix=key))

    def download_files(self, remote_path, local_dir, num_threads=8, quite_mode=True):
        """
    Downloads the files from s3 to  local directory
        :param quite_mode: If False, prints the status of each file downloaded
        :param remote_path: The remote s3 path prefix
        :param local_dir: The local directory
        :param num_threads: The number of parallel downloads

        """
        lp = lambda b, k, r, l: os.path.join(l, *("s3://{}/{}".format(b, k).replace(r, "").split("/")[0:-1]))
        input_tuples = (
            ("s3://{}/{}".format(s3_bucket, s3_key), lp(s3_bucket, s3_key, remote_path, local_dir), quite_mode) for
            s3_bucket, s3_key in
            self.list_files(remote_path))

        with ThreadPool(num_threads) as pool:
            pool.starmap(self.download_file, input_tuples)

    def download_objects(self, s3_prefix, num_threads=8):
        """
Downloads stream of S3 objects, without saving into the local disk
        :param s3_prefix: The s3 prefix, e.g. s3://mybucket/prefix/
        :param num_threads: The number of threads to use
        :return: A list of byte arrays
        """
        s3_files = ("s3://{}/{}".format(s3_bucket, s3_key) for s3_bucket, s3_key in self.list_files(s3_prefix))

        with ThreadPool(num_threads) as pool:
            results = pool.map(self.download_object, s3_files)

        return results

    def upload_files(self, local_dir, remote_path, quite_mode=True, num_workers=os.cpu_count() - 1):
        """
Uploads the files in local directory to s3
        :param quite_mode: If False, prints the status of each file downloaded
        :param num_workers: The number of multi-processes to use
        :param local_dir: The local directory
        :param remote_path: The remote s3 prefix
        """
        rp = lambda f, r, l: "{}/{}".format(r.rstrip("/"), "/".join(
            os.path.expandvars(f).lstrip(os.path.expandvars(l)).lstrip(os.path.sep).split(os.path.sep)))

        input_tuples = [(f, rp(f, remote_path, local_dir), quite_mode) for f in
                        glob.glob("{}/**".format(local_dir), recursive=True) if os.path.isfile(f)]

        partition_size = int(math.ceil(len(input_tuples) / num_workers))
        partitioned_input_tuples = [input_tuples[i:i + partition_size] for i in
                                    range(0, len(input_tuples), partition_size)]

        with Pool(max(1, num_workers)) as processpool:
            processpool.map(self._uploadfiles_multi_thread, partitioned_input_tuples)

    def _uploadfiles_multi_thread(self, input_tuples, num_threads=8):
        """
Uploads files using a multi threaded pool
        :param input_tuples: The input tuples format ( local_file, s3_file, quite_mode)
        :param num_threads: The number of threads
        """
        with ThreadPool(num_threads) as pool:
            pool.starmap(self.upload_file, input_tuples)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("s3url",
                        help="The s3 path. to download from e.g. s3://mybuck/prefix/")

    parser.add_argument("localdir",
                        help="The local directory to save the file to")

    parser.add_argument("--quiet",
                        help="Quiet mode on", type=int, default=1, choices={0, 1})

    args = parser.parse_args()

    print("Starting download..")
    start_time = datetime.datetime.now()

    s3_util = S3Util()
    s3_util.download_files(args.s3url, args.localdir, quite_mode=args.quiet)

    download_endtime = datetime.datetime.now() - start_time

    print("Total time in seconds to download  {} ".format(download_endtime.total_seconds()))
