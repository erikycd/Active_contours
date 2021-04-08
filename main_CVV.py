# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:43:10 2021

@author: Erik
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import cv2 as cv
import math


class ChanVeseVectorial():
    
    def __init__(self, Image = None, mu = None, lambda_in = None, lambda_out = None, 
                 learning_rate = None, stopping_term = None):
        self.__Image = Image
        self.__mu = mu
        self.__lambda_in = lambda_in
        self.__lambda_out = lambda_out
        self.__learning_rate = learning_rate
        self.__stopping_term = stopping_term
        self.__eps = 0.00000001
        self.__epsilon = 10
        self.__nu = 0
        
    def Image(self):
        return self.__Image

    def mu(self):
        return self.__mu

    def lambda_in(self):
        return self.__lambda_in

    def lambda_out(self):
        return self.__lambda_out

    def learning_rate(self):
        return self.__learning_rate

    def stopping_term(self):
        return self.__stopping_term
    
    def mask_generator(self):
        height, width = self.__Image.shape[0], self.__Image.shape[1]
        self.__mask = np.zeros((height, width, 3), np.uint8)
        centre = (int(width/2) , int(height/2))
        radius = 50
        cv.circle(self.__mask, centre, radius , (255,255,255), -1)
        return self.__mask
    
    def see_mask(self):
        plt.figure()
        plt.imshow(self.__mask)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    def distance_map(self):
        phi_temp = self.__mask[:,:,0]
        phi_bin_temp = phi_temp.astype('bool').astype('uint8')
        dist_sum = cv.distanceTransform(phi_bin_temp, cv.DIST_L2, 5);
        dist_dif = cv.distanceTransform(1-phi_bin_temp, cv.DIST_L2, 5);
        self.__phi_dist = dist_sum - dist_dif
        return self.__phi_dist
    
    def see_map(self):
        plt.figure()
        plt.imshow(self.__phi_dist)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    def compute(self):
        learning_rate = self.__learning_rate
        stopping_term = self.__stopping_term
        
        height, width, layers = self.__Image.shape[0], self.__Image.shape[1], self.__Image.shape[2]
        
        vector = np.arange(0, stopping_term, learning_rate)
        segmentation_frame = np.empty((len(vector), height, width), dtype='float32') 
        
        initialization = 1.*(self.__phi_dist > 0)
        segmentation_frame[0] = initialization
        
        phi = self.__phi_dist
        
        for i, dt in zip(range(1, len(vector)), vector):
        
            #%% LENGTH TERM
            
            rows, columns = height, width
            
            # Phi_x = (Phi(:,[2:n,n]) - Phi(:,[1,1:n-1]))/2;
            phi_x1 = np.column_stack((phi[:,1:columns] , phi[:,columns-1]))
            phi_x2 = np.column_stack((phi[:,0] , phi[:,0:columns-1]))
            phi_x = (phi_x1 - phi_x2)/2
            
            #Phi_y = (Phi([2:m,m],:) - Phi([1,1:m-1],:))/2;
            phi_y1 = np.vstack(( phi[1:rows,:] , phi[rows-1,:]))
            phi_y2 = np.vstack(( phi[0,:] , phi[0:rows-1,:]))
            phi_y = (phi_y1 - phi_y2)/2
            
            # Phi_xx = Phi(:,[2:n,n]) - 2*Phi + Phi(:,[1,1:n-1]);
            # Phi_yy = Phi([2:m,m],:) - 2*Phi + Phi([1,1:m-1],:);
            phi_xx = phi_x1 - 2*phi + phi_x2
            phi_yy = phi_y1 - 2*phi + phi_y2
            
            # Phi([2:m,m],[2:n,n])
            phi_xy1 = np.column_stack((phi_y1[:,1:columns] , phi_y1[:,columns-1]))
            
            # Phi([1,1:m-1],[1,1:n-1])
            phi_xy2 = np.column_stack((phi_y2[:,0] , phi_y2[:,0:columns-1]))
            
            # Phi([1,1:m-1],[2:n,n])
            phi_xy3 = np.column_stack((phi_y2[:,1:columns] , phi_y2[:,columns-1]))
            
            # Phi([2:m,m],[1,1:n-1])
            phi_xy4 = np.column_stack((phi_y1[:,0] , phi_y1[:,0:columns-1]))
            
            # Phi_xy = ( Phi([2:m,m],[2:n,n]) + Phi([1,1:m-1],[1,1:n-1]) - Phi([1,1:m-1],[2:n,n]) - Phi([2:m,m],[1,1:n-1]) ) / 4;
            phi_xy = (phi_xy1 + phi_xy2 - phi_xy3 - phi_xy4)/4
            
            #Num = Phi_xx.*Phi_y.^2 - 2*Phi_x.*Phi_y.*Phi_xy + Phi_yy.*Phi_x.^2;
            #Den = (Phi_x.^2 + Phi_y.^2).^(3/2) + a;
            #Curvature = Num./Den;
            
            eps = self.__eps
            Num = phi_xx*np.square(phi_y) - 2*phi_x*phi_y*phi_xy + phi_yy*np.square(phi_x)
            Den = np.power((np.square(phi_x) + np.square(phi_y)) , (3/2)) + eps
            kappa = Num / Den
            
            #%% REGION TERM
            
            lambda_in = self.__lambda_in
            lambda_out = self.__lambda_out
            
            phi_bin = 1.*(phi>0)
            force_image = np.empty((rows,columns,layers),dtype='float32')
            
            for channel in range(0,layers):
                
                img_1 = self.__Image[:,:,channel]
                c_in = np.sum(img_1 * phi_bin) / len(np.argwhere(phi_bin > 0))
                c_out = np.sum(img_1 *(1 - phi_bin)) / len(np.argwhere(phi_bin == 0))
                
                force_image[:,:,channel] = (-1)*lambda_in*np.square(img_1 - c_in) + lambda_out*np.square(img_1 - c_out)
            
            force_image_norm = (1/layers)*np.sum(force_image, axis = 2)
            
            #%% TOTAL FORCE
            
            mu = self.__mu
            nu = self.__nu
            epsilon  = self.__epsilon
            
            delta_epsilon = dt*epsilon/(math.pi*(np.square(epsilon) + np.square(phi)))
            
            phi = phi + delta_epsilon*(mu*kappa - nu + force_image_norm)
            
            #%% PLOT
            
            phi_bin = 1.*(phi>0)
            segmentation_frame[i] = phi_bin
            
            print('Step: ', [i] , [dt])
            
            plt.close("all")
            plt.imshow(cv.cvtColor(self.__Image, cv.COLOR_BGR2RGB))
            plt.xticks([])
            plt.yticks([])
            plt.contour(phi_bin)
            plt.show(block=False)
            plt.pause(0.2)
            
        self.__result = phi_bin
        return self.__result
    
    def display_result(self):
        plt.figure()
        plt.imshow(cv.cvtColor(self.__Image, cv.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.contour(self.__result)
        plt.show()
    

def main():
    Path_image = 'C:/Users/Erik/Google Drive/Computer_Programs/ActiveContours/Erika 2015/airplane.jpg'
    Sample_image = cv.imread(Path_image)
    
    plt.figure()
    plt.imshow(cv.cvtColor(Sample_image, cv.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    mu = 0.1
    lambda_in = 1
    lambda_out = 1
    learning_rate = 0.1
    stopping_term = 1.1
    
    CVV = ChanVeseVectorial(Sample_image, mu, lambda_in, lambda_out, learning_rate, stopping_term)
    
    CVV.mask_generator()
    CVV.see_mask()
    
    CVV.distance_map()
    CVV.see_map()
    
    CVV.compute()    
    CVV.display_result()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    