---
title: HyperLogLog估计算法模拟
date: 2019-11-14
copyright: true
categories: English
tags: [Redis,Spark,distributed compute,Big Data,Estimate]
mathjax: false
mathjax2: false
toc: true
---

## 巨人的肩膀

1. [论文《HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm》](https://links.jianshu.com/go?to=http%3A%2F%2Falgo.inria.fr%2Fflajolet%2FPublications%2FFlFuGaMe07.pdf)
2. 钱文品《Redis深度历险：核心原理与应用实践》
3. [探索HyperLogLog算法（含Java实现）](https://www.jianshu.com/p/55defda6dcd2)
4. [HyperLogLog 算法的原理讲解以及 Redis 是如何应用它的](https://juejin.im/post/5c7900bf518825407c7eafd0)
5. [Sketch of the Day: HyperLogLog — Cornerstone of a Big Data Infrastructure](https://research.neustar.biz/2012/10/25/sketch-of-the-day-hyperloglog-cornerstone-of-a-big-data-infrastructure/)
6. [Redis new data structure: the HyperLogLog](http://antirez.com/news/75)
7. [HyperLogLog 算法的原理讲解以及 Redis 是如何应用它的](https://juejin.im/post/5c7900bf518825407c7eafd0)
8. [走近源码：神奇的HyperLogLog](https://zhuanlan.zhihu.com/p/58519480)

## 源码模拟试验

**说明**：不考虑论文中的常数无偏估计修正因子，以尽量简单的方式模拟，而且从结果中可以看出分桶数据（数组）大小会影响HyperLogLog算法的精确度，也因此反过来可以理解常数无偏估计因子需要根据分桶数据（数组）大小适时调整。

```java
import java.util.concurrent.ThreadLocalRandom;

public class HyperLogLogTest {
    static class BitBucket {

        /**
         * 桶内所有随机数的二进制形式的低位零的最大个数
         */
        private byte maxbits;

        /**
         * 求解出在这个桶内所有随机数的二进制形式的低位零的最大个数
         *
         * @param value 这个桶内的任意一个随机数
         */
        public void update(long value) {
            byte bits = lowZeros(value);
            //更新这个桶的低位零的最大个数，所以maxbits只能估计这个桶内随机数的数量，注意多个桶
            this.maxbits = bits > this.maxbits ? bits : this.maxbits;
        }

        /**
         * 求解某个数它的二进制形式的低位0的个数
         *
         * @param value 随机数
         * @return 低位0的个数
         */
        private byte lowZeros(long value) {
            byte i = 1;
            for (; i < 32; i++) {
                if (value >> i << i != value) {
                    break;
                }
            }
            return (byte)(i-1);
        }
    }


    static class Experiment {
        private int trialNum;
        private short bucketNum;
        private BitBucket[] bitBuckets;

        public Experiment(int trialNum, short bucketNum) {
            this.trialNum = trialNum;
            this.bucketNum = bucketNum;
            this.bitBuckets = new BitBucket[bucketNum];
            for (int i = 0; i < bucketNum; i++) {
                this.bitBuckets[i] = new BitBucket();
            }
        }

        public void work() {
            for (int i = 0; i < this.trialNum; i++) {
                //获取一个随机数，业务上面可以假想成：UUID，活动记录等等的hash值
                //重要的是满足随机性，这里用随机数模拟才能保证HyperLogLog算法的有效性和精确度
                long m = ThreadLocalRandom.current().nextLong(1L << 32);
                //获取这个随机数对应的桶
                BitBucket bitBucket = bitBuckets[(int) (((m & 0xffff0000) >> 16) % this.bucketNum)];
                //更新这个桶的所有随机数的二进制形式的低位0的最大个数
                bitBucket.update(m);
            }
        }

        /**
         * 利用HyperLogLog的算法对实验次数的估计值
         *
         * @return 估计值
         */
        public double estimate() {
            double maxbitInverseSum = 0.0;
            for (BitBucket bitBucket : bitBuckets) {
                maxbitInverseSum += 1.0 / (float) bitBucket.maxbits;
            }
            double harmonicMean = this.bucketNum / maxbitInverseSum;
            return Math.pow(2, harmonicMean) * this.bucketNum;
        }
    }

    public static void main(String[] args) {
        short bukectNum;
        double harmonicMean;
        double estimateErrorRate;
        double estimateErrorRateInverse;
        double estimateErrorRateAvg;
        double est = 0.0;
        for (byte j = 14; j >= 6; j--) {
            bukectNum = (short) Math.pow(2, j);
            estimateErrorRateAvg = 0.0;
            estimateErrorRateInverse = 0.0;
            for (int i = 200000; i <= 2000000; i += 200000) {
                Experiment exp = new Experiment(i, bukectNum);
                exp.work();
                est = exp.estimate();
                estimateErrorRate = Math.abs(est - i) / i;
                estimateErrorRateAvg += estimateErrorRate;
                estimateErrorRateInverse += 1 / estimateErrorRate;
                System.out.printf("%d %.3f %.3f\n", i, est, estimateErrorRate);
            }
            harmonicMean = 10 / estimateErrorRateInverse;
            estimateErrorRateAvg /= 10;
            System.out.println("桶数：" + bukectNum + ", 平均：" + estimateErrorRateAvg + ", 调和平均：" + harmonicMean + "\n");
        }
    }
}

```

### 试验结果

```shell
200000 16384.000 0.918 #注意：桶内随机数太少，不满足随机性和足够的样本容量
400000 319525.334 0.201
600000 505962.682 0.157
800000 698289.180 0.127
1000000 900077.739 0.100
1200000 1087621.839 0.094 #从这个样本容量开始，满足抽样的随机性，HyperLogLog算法得以发挥
1400000 1304437.661 0.068
1600000 1469819.103 0.081
1800000 1669666.816 0.072
2000000 1845353.676 0.077
桶数：16384, 平均：0.18960571419843567, 调和平均：0.10562357091733171

200000 159352.060 0.203
400000 348148.548 0.130
600000 541721.720 0.097
800000 737661.410 0.078
1000000 942261.594 0.058
1200000 1141143.750 0.049
1400000 1361564.781 0.027
1600000 1537696.500 0.039
1800000 1692330.506 0.060
2000000 1968427.770 0.016
桶数：8192, 平均：0.07567032338037119, 调和平均：0.04637322622929935

200000 177238.898 0.114
400000 364321.245 0.089
600000 562224.082 0.063
800000 775783.867 0.030
1000000 1021759.320 0.022
1200000 1154526.053 0.038
1400000 1338244.486 0.044
1600000 1604504.350 0.003
1800000 1778948.301 0.012
2000000 1987024.401 0.006
桶数：4096, 平均：0.0420996187985746, 调和平均：0.013178971576605327

200000 180307.864 0.098
400000 390080.489 0.025
600000 566246.137 0.056
800000 817751.889 0.022
1000000 986946.287 0.013
1200000 1187760.915 0.010
1400000 1430421.433 0.022
1600000 1605588.554 0.003
1800000 1794603.571 0.003
2000000 2112011.625 0.056
桶数：2048, 平均：0.03091849770917094, 调和平均：0.01029119723250593

200000 194312.848 0.028
400000 386648.893 0.033
600000 576056.608 0.040
800000 828576.160 0.036
1000000 1011162.106 0.011
1200000 1285425.863 0.071
1400000 1440170.191 0.029
1600000 1727461.010 0.080
1800000 1963140.456 0.091
2000000 2077138.318 0.039
桶数：1024, 平均：0.04573485746225806, 调和平均：0.03266019810377191

200000 205210.301 0.026
400000 438146.356 0.095
600000 603954.951 0.007
800000 753863.379 0.058
1000000 967701.597 0.032
1200000 1257909.108 0.048
1400000 1502659.824 0.073
1600000 1619446.014 0.012
1800000 1747104.882 0.029
2000000 2270369.251 0.135
桶数：512, 平均：0.05162887564250197, 调和平均：0.02457325664618657

200000 191260.947 0.044
400000 438321.676 0.096
600000 558098.091 0.070
800000 836355.100 0.045
1000000 1107291.478 0.107
1200000 1248805.971 0.041
1400000 1565138.959 0.118
1600000 1776091.504 0.110
1800000 1943602.042 0.080
2000000 2325410.231 0.163
桶数：256, 平均：0.08732405839292125, 调和平均：0.07153063739557493

200000 201052.706 0.005
400000 447490.927 0.119
600000 503101.620 0.161
800000 799865.427 0.000
1000000 944046.078 0.056
1200000 1230985.985 0.026
1400000 1356999.132 0.031
1600000 1694042.415 0.059
1800000 1825287.933 0.014
2000000 2206206.744 0.103
桶数：128, 平均：0.05740755761168429, 调和平均：0.0015781846255803417

200000 234332.476 0.172
400000 370402.926 0.074
600000 786417.984 0.311
800000 705595.987 0.118
1000000 1236640.933 0.237
1200000 1063361.567 0.114
1400000 1487164.878 0.062
1600000 1936580.851 0.210
1800000 1792995.043 0.004
2000000 2178572.682 0.089
桶数：64, 平均：0.13906646547273657, 调和平均：0.030028482076914373
```



