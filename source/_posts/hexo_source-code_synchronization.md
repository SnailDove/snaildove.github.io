---
title: 同步hexo博客
date: 2018-02-10
copyright: true
---

### 旧电脑上博客根目录执行
```shell
git init
git check -b source
git check source
git add *
git rm --cached ./node_modules/ ./source/_drafts ._config.yml ./themes/next/_config.yml
vim .gitignore #添加：./node_modules/ ./source/_drafts ._config.yml （博客根目录配置文件，防止泄露隐私）./themes/next/_config.yml (主题配置文件，防止泄露隐私，百度云盘隐藏空间备份)
git commit -m "初次同步博客"
git push origin source:source
```
### 新电脑上博客的新目录下执行

git 配置完成以后
```shell
git clone https://github.com/~/~.github.io.git # ~代表你的github用户名
cd ~.github.io.git
npm install hexo
npm install
npm install hexo-deployer-git
```
### latex公式问题 
按网上的教程继续修改 `./node_modules/marked/marked.js` 以支持数学公式中的 `\` `-` 