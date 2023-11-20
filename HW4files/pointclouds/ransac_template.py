#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
###YOUR IMPORTS HERE###
###YOUR IMPORTS HERE###
class RANSACClass():
    def __init__(self, error_thres = 0.2 , inlinier_thres = 100):
        self.coeff_best = None
        self.error_thres = error_thres
        self.inliners_thres = inlinier_thres
        
    def calculateNumConsensus(self,points, coeff, thres, b = 1):
        num = 0
        in_idx = [] 
        out_idx = []
        for i, point in enumerate(points):
            error = self.calculateErrror(point,coeff,b)
            if error < thres:
                num +=1
                in_idx.append(i)
            else:
                out_idx.append(i)
        return num, in_idx, out_idx
    
    def calculateConsensus(self,points, coeff, thres, b = 1):
        in_set = [] 
        out_set = []
        error_all = 0
        for point in points:
            error = self.calculateErrror(point,coeff,b)
            error_all +=  error
            if error < thres:
                in_set.append(point)
            else:
                out_set.append(point)
        return error_all, in_set, out_set

    def leastSquare(self,points):
        gamma = 0.0001
        A = np.transpose(np.hstack(points))
        b = -np.ones((len(points),1))
        coeffs = np.linalg.inv(A.T @ A + gamma*np.eye(points[0].shape[0])) @ A.T @ b
        return coeffs
    
    def calculateErrror(self, point, coeff, b = 1):
        return float(abs((coeff.T @ point + b))/np.linalg.norm(coeff))

    def calculateErrors(self,points, coeff):
        error = 0
        for point in points:
            error += self.calculateErrror(point,coeff)
        return error
    
    def run(self,pc, num_iters, num_inliers):
        
        inliners_set = []
        outliners_set = []
        error_best = 1000
        
        for i in tqdm(range(num_iters)):
            np.random.shuffle(pc)
            inliners = pc[0:num_inliers]
            outliners = pc[num_inliers:]
            
            coeffs = self.leastSquare(inliners)
            
            num_consensus, inliners_idxs, outliners_idxs = self.calculateNumConsensus(outliners,coeffs,self.error_thres)
            if num_consensus < self.inliners_thres:
                continue
            
            inliners_points = inliners + [outliners[i] for i in inliners_idxs]
            coeffs = self.leastSquare(inliners_points)
            fit_error = self.calculateErrors(inliners_points,coeffs)
            
            if fit_error < error_best:
                error_best = fit_error
                self.coeff_best = np.vstack((coeffs/np.linalg.norm(coeffs),1/np.linalg.norm(coeffs)))
                inliners_set = [outliners[i] for i in inliners_idxs]
                outliners_set = [outliners[i] for i in outliners_idxs]
                
        return inliners_set, outliners_set
        
        

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    fig1 = utils.view_pc([pc])
    #Fit a plane to the data using ransac
    
    ransac = RANSACClass()
    inliners_set, outliners_set = ransac.run(pc, num_iters=100, num_inliers=30)
            

    #Show the resulting point cloud

    #Draw the fitted plane
    print("Coefficients: {}".format(ransac.coeff_best.T))
    fig2 = utils.view_pc([inliners_set],color='r')
    fig2 = utils.view_pc([outliners_set],fig2,color='b')
    
    utils.draw_plane(fig2, ransac.coeff_best[:3], -ransac.coeff_best[:3]*ransac.coeff_best[3], color=(0.5, 0.9, 0.01, 0.5), length=[-1, 1], width=[-1, 1])


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
