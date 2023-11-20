#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from ransac_template import RANSACClass
from pca_template import PCAClass
###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def plotPlane(inliners_set, outliners_set, coeff):
    fig = utils.view_pc([inliners_set],color='r')
    fig = utils.view_pc([outliners_set],fig,color='b')
    
    utils.draw_plane(fig, coeff[:3], -coeff[:3]*coeff[3], color=(0.5, 0.9, 0.01, 0.5), length=[-1, 1], width=[-1, 1])
    return fig

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')
    error_thres = 0.2
    num_tests = 10
    fig = None
    fig = utils.view_pc([pc])
    
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test

        ###YOUR CODE HERE###
        pca = PCAClass()
        pca_plane = pca.fit(pc)
        print("PCA plane: {}".format(pca_plane.T))
        ransac = RANSACClass()
        ransac.run(pc, num_iters=100, num_inliers=5)
        ransac_plane = ransac.coeff_best
        print("RANSAC plane: {}".format(ransac_plane.T))
        pca_error, pca_inliners, pca_outliners = ransac.calculateConsensus(pc, pca_plane[:3], thres= error_thres, b = pca_plane[-1])  
        ransac_error, ransac_inliners, ransac_outliners = ransac.calculateConsensus(pc, ransac_plane[:3], thres= error_thres, b = ransac_plane[-1])  
        #this code is just for viewing, you can remove or change it
        print("PCA error {} vs RANSAC error {}".format(pca_error, ransac_error))
        fig1 = plotPlane(pca_inliners, pca_outliners, pca_plane)
        fig2 = plotPlane(ransac_inliners, ransac_outliners, ransac_plane)
        input("Press enter for next test:")
    
        plt.close(fig1)
        plt.close(fig2)
        ###YOUR CODE HERE###
    plt.close(fig)
    input("Press enter to end")


if __name__ == '__main__':
    main()
