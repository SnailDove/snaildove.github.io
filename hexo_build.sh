
#step1：初始化博客
echo "start hexo init" && hexo init && echo "init finish" && 
echo "start cnpm install" && cnpm install && echo "cnpm install finish" && 
cnpm install  hexo-deployer-git --save && # 安装git部署器（上传网页代码到github.com）

#step2: 安装主题
git clone https://github.com/tufu9441/maupassant-hexo.git themes/maupassant && 
cnpm install hexo-renderer-pug --save && cnpm install hexo-renderer-sass --save && 
#step3：配置插件
cnpm install hexo-generator-search --save && #站内搜索插件
cnpm install hexo-generator-feed --save && #feed 插件
cnpm install hexo-helper-qrcode --save && #二维码分享插件

#step4: #marked.js与mathjax语义冲突
#修改博客根目录下的 /node_modules/marked/lib/rules/inline.js 文件：
# //escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,      将其修改为
# escape: /^\\([`*\[\]()#$+\-.!_>])/,
# //em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,    第20行，将其修改为
# em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,

#step5: 站点地图（sitemap.xml）
cnpm install hexo-generator-sitemap --save &&
cnpm install hexo-generator-baidu-sitemap --save &&

#step6: 微博秀
#http://blog.csdn.net/u010053344/article/details/50757597 #测试问题
#http://blog.csdn.net/yongf2014/article/details/50015001

#step7: GFW
#https://www.jianshu.com/p/94500314e400

#step8: 不算子
#http://ibruce.info/2015/04/04/busuanzi/
#在index.pug页，增加如下代码：
 #     .post-meta= post.date.format(config.date_format)
 #       if theme.busuanzi == true
 #         script(src='https://dn-lbstatics.qbox.me/busuanzi/2.3/busuanzi.pure.mini.js', async)
 #         span#busuanzi_container_page_pv= ' | '
 #           span#busuanzi_value_page_pv
 #           span= ' ' + __('Hits ')
#配置主题文件开启不算子
busuanzi: true

#step9: 标签云安装
cnpm install hexo-tag-cloud@^2.0.* --save


#step10: 安装文章加密
cnpm install --save hexo-blog-encrypt

#去掉 主题目录下: /next/layout/_partials/head.swig文件中第一个{% endif %}前的:
# <script>
    # (function(){
        # if('{{ page.password }}'){
            # if (prompt('please input password') !== '{{ page.password }}'){
                # alert('false password！');
                # history.back();
            # }
        # }
    # })();
# </script>
