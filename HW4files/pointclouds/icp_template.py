#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from sklearn.neighbors import KDTree
from tqdm import tqdm
import sys
###YOUR IMPORTS HERE###

class ICPClass():
    def __init__(self):
        self.num_iters = 100
        self.error_thres = 0.1
    def fit(self,sourcePointClouds):
        leaf_size = 30
        self.sourcePC = sourcePointClouds
        self.kdTree = KDTree(sourcePointClouds,leaf_size=leaf_size, metric='euclidean')
        
    def findCorrespondences(self,pc):
        num_corr = 1
        idx = self.kdTree.query(pc,num_corr,return_distance=False)
        idx = list(idx.reshape(-1))
        return self.sourcePC[idx,:]
    
    def optimizeCorrespondences(self,source_pc, target_pc):
        p_mean = np.mean(source_pc,axis=0).reshape((-1,1))
        q_mean = np.mean(target_pc, axis = 0).reshape((-1,1))
        
        X = source_pc.T  - p_mean
        Y = target_pc.T - q_mean
        
        S = X @ Y.T
        
        U, D, Vh = np.linalg.svd(S)
        
        gamma = np.linalg.det(np.dot(U, Vh).T)
        R = Vh.T @ np.diag([1, 1, gamma]) @ U.T
        t = q_mean - R @ p_mean
        
        return R, t
    
    def applyTransform(self,pc, R, t):
        return np.transpose(R @ pc.T + t)
    
    def calculateErrors(self, sourcePC, targetPC):
        return np.sum(np.square(sourcePC - targetPC))
        
        
    def computeTransform(self, targetPointClouds):
        
        R  = np.eye(3)
        t = np.zeros((3,1))
        next_pc = targetPointClouds.copy()
        
        for i in tqdm(range(self.num_iters)):
            corr_points = self.findCorrespondences(next_pc)
            R, t = self.optimizeCorrespondences(next_pc,corr_points)
            next_pc = self.applyTransform(next_pc,R,t)
            error = self.calculateErrors(next_pc,corr_points)
            if abs(error) < self.error_thres:
                break
        return R, t
        


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target2.csv' if len(sys.argv) ==1 else 'cloud_icp_target' + sys.argv[1]+'.csv') # Change this to load in a different target



    fig = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###
    icp = ICPClass()
    pc_source_arr = np.asarray(np.hstack(pc_source)).T
    pc_target_arr = np.asarray(np.hstack(pc_target)).T
    icp.fit(pc_source_arr)
    R, t = icp.computeTransform(pc_target_arr)
    print("Transformation matrix \n{}".format(np.vstack((np.hstack((R, t)), [0, 0, 0 ,1]))))
    
    result = icp.applyTransform(pc_target_arr,R,t)
    result = [result[i].reshape((-1,1)) for i in range(result.shape[0])]
   
    fig = utils.view_pc([result], fig, 'y','*')
    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
