#-*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))
def randCent(dataSet,k):
    #维度
    n = shape(dataSet)[1]
    #构建随机簇心
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,creataCent=randCent):
    m = shape(dataSet)[0]   #记录总数
    clusterAssment = mat(zeros((m,2)))
    centroids = creataCent(dataSet,k)   #初始质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            #寻找最近质心
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    #minDist = distJI
                    minDist = distJI
                    minIndex = j

            #当样本i的最小距离质心没有发生变化时，停止迭代
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            #记录第i个数据属于哪个质心，并且记录距离
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        #更新质心的位置
        for cent in range(k):
            #找出属于同一类的样本
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            #重新计算该类样本的质心
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]   #记录总数
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0] #创建初始簇的质心（1个）
    centList = [centroid0]
    print shape(centList)
    #计算SSE sum of squared error
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j:,1])**2
    while len(centList)<k:
        lowestSSE = inf
        #遍历每个质心
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #针对每个簇进行二分类
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            print shape(centroidMat)
            sseSplit = sum(splitClustAss[:,1]) #全部误差平方和SSE
            #其他类的误差平方和的总和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit and sseNotSplit:",sseSplit,sseNotSplit
            #如果分割后的误差总和小于原先总误差,新增簇
            if(sseSplit+sseNotSplit)<lowestSSE:
                bestCentToSplit = i #标记最好划分簇
                bestNewCents = centroidMat #新划分的2个质心
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:]) #新增新的质心
        print bestNewCents
        print bestNewCents[1,:]
        print centList[0]
        print centList
        print shape(centList)
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
    print shape(centList)
    return centList,clusterAssment


if __name__ == "__main__":
    data = mat(loadDataSet("testSet2.txt"))

    #myCentroids,clustAsing = kMeans(data,3)
    #plt.scatter(data[:,0],data[:,1],s=50,c=clustAsing[:,0].A)
    # plt.scatter(myCentroids[:,0],myCentroids[:,1],s=120,c='red',marker='*')
    # plt.show()

    centList,myNewAssments = biKmeans(data,3)
    plt.scatter(data[:,0],data[:,1],s=50,c=myNewAssments[:,0].A)
    #plt.scatter(myNewAssments[:,0],myNewAssments[:,1],s=120,c='red',marker='*')
    plt.show()

