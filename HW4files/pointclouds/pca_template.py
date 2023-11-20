#!/usr/bin/env python
import utils
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from utils import convert_pc_to_matrix
###YOUR IMPORTS HERE###

class PCAClass():
    
    def __init__(self):      
        self.principal_vectors = None
        
    def fit(self, pc):
        n = len(pc)
        self.mu = convert_pc_to_matrix(pc).mean(axis = 1)
        Xh = convert_pc_to_matrix(pc) - self.mu
        Q = 1/(n-1)* (Xh @ Xh.T)
        U,D,Vh = np.linalg.svd(Q)
        self.principal_vectors = Vh.T
        self.eigens = D
        return np.vstack((self.principal_vectors[:,-1],-self.principal_vectors[:,-1].T@ self.mu))
        
    def rotate(self,pc):
        new_pc = []
        for point in pc:
            new_pc.append(self.principal_vectors.T @ point)
        return new_pc
    
    def rotateNoise(self,pc, thres=0.01):
        reduced_principal_vectors = self.principal_vectors.copy()
        new_pc = []
        for i in range(self.eigens.shape[0]):
            if self.eigens[i]**2 < thres:
                reduced_principal_vectors[:,i] = 0
        for point in pc:
            new_pc.append(reduced_principal_vectors.T @ point)
        
        return new_pc
        

def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    thres = 0.001
    # Show the input point cloud
    fig = utils.view_pc([pc])
    
    pca = PCAClass()
    pca.fit(pc)
    
    #Rotate the points to align with the XY plane
    new_pc = pca.rotate(pc)

    normal_vec = pca.principal_vectors[:,-1].copy()
 
    utils.draw_plane(fig, normal_vec, pca.mu, color=(0.01, 0.9, 0.01, 0.5), length=[-1, 1], width=[-1, 1])

    
    #Show the resulting point cloud
    fig1 = utils.view_pc([new_pc])

    #Rotate the points to align with the XY plane AND eliminate the noise
    
    new_pc = pca.rotateNoise(pc,0.001)
    # Show the resulting point cloud
    fig2 = utils.view_pc([new_pc])
    ###YOUR CODE HERE###


    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
