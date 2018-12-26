#!/bin/bash

##
# Step1: qshell account QRnBlzUs_tP4nUgMfpF1l_ZhDCwpJvg1zGcTO_hS vD_sH9xL3-DZ5KcbLtKXmYrVpclGIJhyciT_6iyf SnailDove
# Step2: qshell buckets 找到自己的上一个bucket
# Step3: qshell listbucket2 old_bucket_name A.list.txt
# Step4: awk "{print $1}" A.list.txt > list.txt
# Step5: 到官网：qiniu.com 创建新的bucket即新的空间： new_bucket_name
# Step6: qshell batchmove old_bucket_name new_bucket_name -i list.txt
# Step7: 将下面第二对#里面内容复制到第一对#里面，然后获取到新域名的第一个点之前内容，复制到下面sed的第2对##里面

file_path='./source/_posts/';
file_names=`ls ${file_path}`;

echo ${file_names}>1.txt


for file in ${file_names}
do
	sed -i 's#pil64vxk1#pkaunwk1s#g' "${file_path}${file}" & 
done;

#https://portal.qiniu.com/cdn/domain/pkaunwk1s.bkt.clouddn.com