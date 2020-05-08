#!/bin/bash


# Step1: qshell buckets 找到自己的上一个bucket, 给下文的old_bucket_name变量赋值
# Step2: 到官网或者七牛客户端：qiniu.com 创建新的bucket即新的空间, 给下文的new_bucket_name变量赋值

current_dir_path=$(cd "$(dirname "$0")";pwd) 
new_bucket_name='snaildove-blog'
old_bucket_name='snaildove-gitpage'

# Step3: 到官网：qiniu.com 找到new bucket 对应的前缀
old_prefix_url='q83p23d9i'
new_prefix_url='q9kvrafcq'

qshell_dir='/Applications'
qshell_exe="${qshell_dir}/qshell"

# 将博客原文档的资源的url变更
blog_local_dir=${current_dir_path}
source_file_path="${blog_local_dir}/source/_posts/";
drafts_file_path="${blog_local_dir}/source/_drafts/";
source_file_names=`ls ${source_file_path}`;
draft_file_names=`ls ${drafts_file_path}`;


replace_expr="s#${old_prefix_url}#${new_prefix_url}#g"

for file in ${source_file_names}
do 
    file_with_path="${source_file_path}${file}"
    sed -i "" ${replace_expr} ${file_with_path}
done;

for file in ${draft_file_names}
do 
    file_with_path="${drafts_file_path}${file}"
    sed -i "" ${replace_expr} ${file_with_path}
done;


# # 由于hexo生产十分缓慢，先生成，在资源移到新的bucket
hexo clean && hexo g && 

# # 登录七牛账号
# ${qshell_exe} account QRnBlzUs_tP4nUgMfpF1l_ZhDCwpJvg1zGcTO_hS vD_sH9xL3-DZ5KcbLtKXmYrVpclGIJhyciT_6iyf SnailDove &&

# # 列出需要变更的资源：图片，视频，音频
${qshell_exe} listbucket2 ${old_bucket_name} | awk 'BEGIN{FS="\t"}{print $1}END{}' > list.txt &&
# # 将资源移到新bucket
${qshell_exe} batchmove ${old_bucket_name} ${new_bucket_name} -i list.txt &&

# # 删除中间文件 和 博客部署
rm list.txt && hexo d



#### 替换 jupter notebook ######

blog_ipynb_dir='/Users/brt/workspace/snaildove.github.io.jupyter-notebook'
ipynb_files=`ls ${blog_ipynb_dir}/*.ipynb`

for file in ${ipynb_files}
do
    sed -i "" "s#${old_prefix_url}#${new_prefix_url}#g" "${file}"
done;

cd ${blog_ipynb_dir}
git add *;
git commit -m "update the qiniu sources";
git push origin master:master

#################################




