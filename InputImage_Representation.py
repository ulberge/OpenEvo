import numpy as np
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io
import random
from random import shuffle
import copy
import time

from DesignPhysics import Design
from PIL import Image, ImageDraw
import cv2

##############################################################
# Erik's Code
##############################################################
# Load an image from the file system in the correct format
def load_image(file_name):
    img = np.float32(cv2.imread(file_name))
    # Remove two channels
    img = np.delete(img, [1, 2], axis=2)
    img = img / 255
    return img.squeeze()

# Given a genotype, return an image as a 2D array of numbers [0, 1]
def getImageForGenotype(genotype):
    img = Image.new('F', (225, 225), 1.0)
    draw = ImageDraw.Draw(img)
    for curve in genotype:
        X, Y = CubicBezierPoints(10, curve)
        for i in range(len(X) - 1):
            draw.line((50 + X[i], 200 - Y[i], 50 + X[i + 1], 200 - Y[i + 1]), fill=0.0, width=1)
    img = np.asarray(img)
    return img

# Given a design and a genotype, give a score of how far from the current
# trajectory the genotyp is
def get_score(design, genotype):
    img = getImageForGenotype(genotype)
    score = design.get_score(img)

    # Adjust score to be proportional to evolution
    score = max(0, (0.3 - score) * 100)

    # Save images for inspection that are below or above a certain threshold for debugging
    # if score > 20:
    #     save_img = img * 255
    #     save_img = save_img.astype(np.uint8)
    #     cv2.imwrite('test.png', save_img)
    # if score < 1:
    #     save_img = img * 255
    #     save_img = save_img.astype(np.uint8)
    #     cv2.imwrite('test1.png', save_img)

    return score
##############################################################
# End Erik's Code
##############################################################

############################################################## FUNCTIONS
def distanceList_Euclidian(points,newPoint):
    """
    points = [[X,Y],[X,Y],...]. newPoint = [X,Y].
    returns = [dist1,dist2,...]
    """
    distList = []
    for point in points:
        summation = (point[0]-newPoint[0])**2 + (point[1]-newPoint[1])**2
        distList.append(summation**(0.5))
    return distList
def KMeansClusters(points,kClusters,MaxRounds,LowArg1,HighArg1,LowArg2,HighArg2):
    """
    points = [[X,Y],[X,Y],...]. Returns dictionary with cluster numbers as keys
    and values = [cluster Center, [points assigned to cluster]], where Cluster Center = [X,Y]
    """
    DictioCluster = {}
    DictioKeys = range(kClusters)
    PointsAsCentroids = random.sample(points, kClusters) #Randomly select K points as initial centroids
    for k in DictioKeys: #Initialize k cluster centers with random values
        #RandomCenter = [random.uniform(LowArg1, HighArg1),random.uniform(LowArg2, HighArg2)]
        RandomCenter = [PointsAsCentroids[k][0],PointsAsCentroids[k][1]]
        DictioCluster[k] = [RandomCenter,[]]
    Counter = 1 #Initialize counter in 1
    Rounds = 0
    while Counter != 0 and Rounds < MaxRounds: #while counter different to 0
        Counter = 0 #Initialize counter for this round. It will pass through the while if clusters unchanged
        for k in DictioKeys: #Empty the cluster points
            DictioCluster[k][1] = []
        for point in points: #assign each point to a cluster
            distList = [] #Initialize the distances for this point
            for k in DictioKeys: #For each cluster
                summation = (point[0]-DictioCluster[k][0][0])**2 + (point[1]-DictioCluster[k][0][1])**2
                distList.append(summation**(0.5))
            ClosestClusteri = distList.index(min(distList)) #index of the closest cluster (the center) to point
            DictioCluster[ClosestClusteri][1].append(point) #assign point to the closest cluster
        for k in DictioKeys:#For each cluster, update center
            PointsCluster =  DictioCluster[k][1]
            if len(PointsCluster) != 0:
                clusterX = 0
                clusterY = 0
                for point in PointsCluster:
                    clusterX += point[0]
                    clusterY += point[1]
                clusterX = clusterX/float(len(PointsCluster)) #calculate the center between the points assigned to the cluster
                clusterY = clusterY/float(len(PointsCluster)) #calculate the center between the points assigned to the cluster
                if clusterX != DictioCluster[k][0][0]:
                    DictioCluster[k][0][0] = clusterX #Update X coordinate of the cluster center
                    Counter += 1
                if clusterY != DictioCluster[k][0][1]:
                    DictioCluster[k][0][1] = clusterY #Update Y coordinate of the cluster center
                    Counter += 1
            else: #no point was assigned to cluster, so re-initialize it randomly
                RandomCenter = [random.uniform(LowArg1, HighArg1),random.uniform(LowArg2, HighArg2)]
                DictioCluster[k][0] = RandomCenter
        Rounds += 1
    print "No. Rounds: "+str(Rounds)
    #LAST UPDATE OF THE CLUSTER POINTS ACCORDING TO THE FINAL CENTERS:
    for k in DictioKeys: #Empty the cluster points
        DictioCluster[k][1] = []
    for point in points: #assign each point to a cluster
        distList = [] #Initialize the distances for this point
        for k in DictioKeys: #For each cluster
            summation = (point[0]-DictioCluster[k][0][0])**2 + (point[1]-DictioCluster[k][0][1])**2
            distList.append(summation**(0.5))
        ClosestClusteri = distList.index(min(distList)) #index of the closest cluster (the center) to point
        DictioCluster[ClosestClusteri][1].append(point) #assign point to the closest cluster
    return DictioCluster # Return dictionary with the cluster number as key and values the center coordinates and the points assigned to it.
def DeltaDensity(points):
    """
    points = [[X,Y],[X,Y],...].
    return [Xrange,Yrange,XDistribution,YDistribution]
    """
    Xlist =[]
    Ylist = []
    for P in points:
        Xlist.append(P[0])
        Ylist.append(P[1])
    Xrange = range(min(Xlist),max(Xlist)+1)
    Yrange = range(min(Ylist),max(Ylist)+1)
    Xdensity = [0] * len(Xrange)
    Ydensity = [0] * len(Yrange)
    for x in Xlist:
        Xdensity[x - min(Xlist)] += 1
    for y in Ylist:
        Ydensity[y - min(Ylist)] += 1
    XDeltaDensity = [1] * len(Xrange)
    YDeltaDensity = [1] * len(Yrange)
    for i in range(1,len(Xrange)-1):
        Delta = np.abs(Xdensity[i+1] - Xdensity[i-1])/2.0 #Centered Finite Difference
        Max = max([Xdensity[i+1],Xdensity[i-1]])
        if Max != 0:
            XDeltaDensity[i] = Delta/float(Max) # Delta in proportion of max in the finite difference
    for i in range(1,len(Yrange)-1):
        Delta = np.abs(Ydensity[i+1] - Ydensity[i-1])/2.0 #Centered Finite Difference
        Max = max([Ydensity[i+1],Ydensity[i-1]])
        if Max != 0:
            YDeltaDensity[i] = Delta/float(Max) # Delta in proportion of max in the finite difference
    # Distribution E [0,1]; [Density * (DeltaDensity+0.1)]/Total
    XDistribution = np.multiply(np.array(Xdensity),np.add(np.array(XDeltaDensity),[0.1]))
    Total = np.sum(XDistribution)
    XDistribution = np.true_divide(XDistribution, Total) # Normalized to [0,1]
    YDistribution = np.multiply(np.array(Ydensity),np.add(np.array(YDeltaDensity),[0.1]))
    Total = np.sum(YDistribution)
    YDistribution = np.true_divide(YDistribution, Total)
    #return [Xrange,Yrange,Xdensity,Ydensity,XDeltaDensity,YDeltaDensity,XDistribution,YDistribution]
    return [Xrange,Yrange,XDistribution,YDistribution]
def CornersDetector(points,Tol_ProportionalDeltaDensity):
    """
    points = [[X,Y],[X,Y],...]. When the proportional change in point density in both X and Y is
    >= Tol_ProportionalDeltaDensity, a potential corner is selected.
    return [[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...]; Fitness=mean(DeltaD_X,DeltaD_Y)
    """
    Corners = []
    Xlist =[]
    Ylist = []
    for P in points:
        Xlist.append(P[0])
        Ylist.append(P[1])
    Xrange = range(min(Xlist),max(Xlist)+1)
    Yrange = range(min(Ylist),max(Ylist)+1)
    Xdensity = [0] * len(Xrange)
    for x in Xlist:
        Xdensity[x - min(Xlist)] += 1
    for i in range(len(Xrange)):
        if i == 0: # First point
            XDeriv = Xdensity[i] #Backward Finite Difference; assume density (i-1) = 0
        elif i == range(len(Xrange))[-1]: # Last point
            XDeriv = Xdensity[i] #Forward Finite Difference; assume density (i+1) = 0
        else:
            XDeriv = np.abs(Xdensity[i] - Xdensity[i-1]) #Backward Finite Difference
        Ref = Xdensity[i]
        if Ref != 0:
            XDeriv = XDeriv/float(Ref) # Delta in proportion
        if XDeriv >= Tol_ProportionalDeltaDensity: # X[i] is Interesting in X, then explore in Y
            # How many Y's in X[i]? = Xdensity[i]
            Ydensity_inXi = [0] * len(Yrange)
            for j in range(len(Xlist)):
                if Xlist[j] == Xrange[i]: # This j point has Xi
                    Y = Ylist[j] # This is the corresponding Y value
                    Ydensity_inXi[Y - min(Ylist)] += 1
            # Now see if there is Interesting derivative in Y:
            for j in range(len(Yrange)):
                if j == 0: # First point
                    YDeriv = Ydensity_inXi[j] #Backward Finite Difference; assume density (j-1) = 0
                    #Max = Ydensity_inXi[j+1]
                elif j == range(len(Yrange))[-1]: # Last point
                    YDeriv = Ydensity_inXi[j] #Forward Finite Difference; assume density (j+1) = 0
                    #Max = Ydensity_inXi[j-1]
                else:
                    YDeriv = np.abs(Ydensity_inXi[j] - Ydensity_inXi[j-1]) #Backward Finite Difference
                    #Max = max([Ydensity_inXi[j+1],Ydensity_inXi[j-1]])
                Ref = Ydensity_inXi[j]
                if Ref != 0:
                    YDeriv = YDeriv/float(Ref) # Delta in proportion
                if YDeriv >= Tol_ProportionalDeltaDensity: # X[i] is Interesting in X, and Y[j] in Y
                    # We found a potential corner!!! :D
                    Corners.append([Xrange[i],Yrange[j],np.mean([XDeriv,YDeriv])]) # Corner: Point X[i],Y[j]
    return Corners
def CubicBezierPoints(n,Points):
    """
    The curve is defined from the 4 points, where e.g. Points=[ [X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster ].
    n points are returned being equidistant in parameter t
    return [[X(t0),X(t1),...],[Y(t0),Y(t1),...]]
    https://www.desmos.com/calculator/cahqdxeshd
    """
    tlist = np.linspace(0,1,n)
    Xlist = []
    Ylist = []
    Point0,Point1,Point2,Point3,cluster = Points
    for t in tlist:
        X = Point0[0]*(1-t)**3 + Point1[0]*3*t*(1-t)**2 + Point2[0]*3*(1-t)*t**2 + Point3[0]*t**3
        Y = Point0[1]*(1-t)**3 + Point1[1]*3*t*(1-t)**2 + Point2[1]*3*(1-t)*t**2 + Point3[1]*t**3
        Xlist.append(X)
        Ylist.append(Y)
    return [Xlist,Ylist]
def RouletteWheelSelection(FitnessList):
    """
    [FitnessList] provides the individuals represented by their fitness.
    returns the index of the individual selected
    """
    #FitnessVals = FitnessList
    #
    FitnessVals = []
    for F in FitnessList:
        FitnessVals.append(F**2) # Fitness has a X^2 effect of probability of selection
    #
    Total = np.sum(FitnessVals)
    FitnessVals = np.true_divide(FitnessVals, Total) # Normalized to [0,1]
    R = random.random()
    i = 0
    Sum = FitnessVals[i]
    while R>Sum:
        i += 1
        Sum  += FitnessVals[i]
    return i
def InitializeIndividual(Ncurves,Dictionary_Clusters):
    """
    For every curve:
    -> Select cluster (first in keys order, then Roulette based on number of points)
    -> Select first End Point P0 (Roulette based on corner fitness)
    -> Calculate distance from other corners to 1st one: farther away within cluster -> Fitter
    -> Select the other end P3 (Roulette based on corner distance fitness)
    -> Interpolate linearly to determine the internal control points P1, P2
    Dictionary_Clusters: keys are cluster indexes, related to
    [Center[X,Y], [points[X,Y] assigned to cluster],corners[[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...] ],
    returns a genotype: [ [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster], [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster],...Ncurves]
    """
    ClusterSizes = []
    UsedPoints_i = []
    for k in Dictionary_Clusters.keys():
        Size = len(Dictionary_Clusters[k][1])
        ClusterSizes.append(Size)
        UsedPoints_i.append([])
    Curves = []
    nKlusters = len(Dictionary_Clusters.keys())
    iCluster = 0
    for c in range(Ncurves): # for every cubic Bezier curve
        # Select cluster:
        if iCluster >= nKlusters:
            iCluster = RouletteWheelSelection(ClusterSizes) # More points -> More likely
        CornersFitness = []
        for c in Dictionary_Clusters[iCluster][2]:
            CornersFitness.append(c[2])
        #Select first End P0:
        Try = 0
        P0_i = RouletteWheelSelection(CornersFitness)
        while P0_i in UsedPoints_i[iCluster] and Try < 2: # Try not to repeat a used point
            P0_i = RouletteWheelSelection(CornersFitness)
            Try += 1
        UsedPoints_i[iCluster].append(P0_i) # count P0_i as used in cluster iCluster
        P0 = [ Dictionary_Clusters[iCluster][2][P0_i][0],Dictionary_Clusters[iCluster][2][P0_i][1] ]
        #Calculate distance from other corners to P0:
        Distances_toP0 = distanceList_Euclidian(Dictionary_Clusters[iCluster][2],P0)
        #Select seconf End P3:
        Try = 0
        P3_i = RouletteWheelSelection(Distances_toP0) # Farther from P0 -> More likely
        while P3_i in UsedPoints_i[iCluster] and Try < 2: # Try not to repeat a used point
            P3_i = RouletteWheelSelection(Distances_toP0)
            Try += 1
        UsedPoints_i[iCluster].append(P3_i) # count P0_i as used in cluster iCluster
        P3 = [ Dictionary_Clusters[iCluster][2][P3_i][0],Dictionary_Clusters[iCluster][2][P3_i][1] ]
        #Define intermediate control points:
        X1 = P0[0] + (P3[0]-P0[0])*0.3
        Y1 = P0[1] + (P3[1]-P0[1])*0.3
        X2 = P3[0] - (P3[0]-P0[0])*0.3
        Y2 = P3[1] - (P3[1]-P0[1])*0.3
        Curves.append([ [P0[0],P0[1]],[X1,Y1],[X2,Y2],[P3[0],P3[1]],iCluster ])
        iCluster += 1
    return Curves
def Fitness_RMSE_approxPerpendicular_cornerFit(NBezierPoints,searchWidh,IndividualGenotype,TargetPoints,Dictio_Clusters):
    """
    -> From the genotype, get the phenotype (NBezierPoints points per Bezier curve)
    -> For every contour Potential corner:
       -> Identify if there are Bezier points within a given XY square of widh searchWidh
       -> If so, calculate and store smallest euclidean distance found
       -> If not, set 2*square diagonal as the Error
    -> Calculate and return the Root-Mean-Squared-Approx Euclidean Error: RMSE1
    -> For every Bezier point:
       -> get an integer range of interest
       -> call target points directly in the range of the cross; calculate the smallest euclidian distance
       -> If not, set 2*square diagonal as the Error
    -> Calculate and return the Root-Mean-Squared-Approx Euclidean Error: RMSE2
    -> return RMSE1 + RMSE2
    IndividualGenotype: genotype: [ [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]], [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]],...Ncurves]
    TargetPoints: [[X,Y],[X,Y],...]
    Dictio_Clusters: keys=cluster indexes, values=
    [Center[X,Y], [points[X,Y] assigned to cluster],corners[[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...] ]
    rerturns the RMSE for Individual
    """
    Distances = [] # one for every target points; the euclidean distance to approx closest Bezier point
    # Get the Bezier points (Phenotype) from the genotype, in a list of Xs and one of Ys:
    BezierXs = [] # [X(t0),X(t1),... NBezierPoints]
    BezierYs = [] # [Y(t0),Y(t1),... NBezierPoints]
    for bezier in IndividualGenotype: # each iter: curve gene [ [X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3] ]
        Bout = CubicBezierPoints(NBezierPoints,bezier) # return [[X(t0),X(t1),...],[Y(t0),Y(t1),...]]
        BezierXs += Bout[0]
        BezierYs += Bout[1]
    # Calculate the approx error for every target potential corner
    Corners = []
    for k in Dictio_Clusters.keys():
        Corners += Dictio_Clusters[k][2]
    countD1 = 0 #***
    for Corner in Corners: #each iter: corners[[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...]
        Distance = 2* searchWidh * np.sqrt(2) # Initialize as the 2*diagonal of the search square
        CX = Corner[0]
        CY = Corner[1]
        Xrange = [CX-searchWidh/2.,CX+searchWidh/2.] # Width range of the Vertical space of the search cross
        Yrange = [CY-searchWidh/2.,CY+searchWidh/2.] # Height range of the Horizontal space of the search cross
        # Find shortest distance to all Bezier points in the Xrange of search:
        for i in range(len(BezierXs)): # Check every Bezier point
            if (BezierXs[i] >= Xrange[0] and BezierXs[i] <= Xrange[1]) and (BezierYs[i] >= Yrange[0] and BezierYs[i] <= Yrange[1]):
                # i Bezier point is within search range
                Euclidean = ( (BezierXs[i]-CX)**2 + (BezierYs[i]-CY)**2 )**(0.5)
                if Euclidean < Distance:
                    Distance = Euclidean # Update Error approximation
        Distances.append(Distance)
        if Distance == 2* searchWidh * np.sqrt(2):
            countD1 += 1
    RMSE1 = 0
    for Error in Distances:
        RMSE1 += Error**2
    RMSE1 = np.sqrt(RMSE1/float(len(Distances)))
    #################################################### Calculate error from the Bezier points
    TargetXs = []
    TargetYs = []
    for Target in TargetPoints: # each iter: [X,Y]
        TX = Target[0] # X of the point we want to fit
        TY = Target[1] # Y of the point we want to fit
        TargetXs.append(TX)
        TargetYs.append(TY)
    Distances = []
    countD2 = 0 #***
    for i in range(len(BezierXs)): # for every Bezier point
        Distance = 2* searchWidh * np.sqrt(2) # Initialize as the 2*diagonal of the search square
        TXs = TargetXs[:]
        TYs = TargetYs[:]
        Bx = BezierXs[i]
        By = BezierYs[i]
        Xrange = range(int(Bx-searchWidh/2.),int(Bx+searchWidh/2.)+1)
        Yrange = range(int(By-searchWidh/2.),int(By+searchWidh/2.)+1)
        # look in the X range of the cross
        for X in Xrange:
            while X in TXs: # a relevant-in-X target point found!
                index = TXs.index(X)
                if TYs[index] in Yrange: # a relevant-in-XY target point found!
                    Euclidean = ( (TXs[index]-Bx)**2 + (TYs[index]-By)**2 )**(0.5)
                    if Euclidean < Distance:
                        Distance = Euclidean # Update Error approximation
                TXs[index] = "used" #This Target point won't be used again for this Bezier point
                TYs[index] = "used"
        """
        for X in Xrange:
            while X in TXs: # a relevant target point found!
                index = TXs.index(X)
                Euclidean = ( (TXs[index]-Bx)**2 + (TYs[index]-By)**2 )**(0.5)
                TXs[index] = "used" #This Target point won't be used again for this Bezier point
                TYs[index] = "used"
                if Euclidean < Distance:
                    Distance = Euclidean # Update Error approximation
        # look in the Y range of the cross
        for Y in Yrange:
            while Y in TYs: # a relevant target point found!
                index = TYs.index(Y)
                Euclidean = ( (TXs[index]-Bx)**2 + (TYs[index]-By)**2 )**(0.5)
                TXs[index] = "used" #This Target point won't be used again for this Bezier point
                TYs[index] = "used"
                if Euclidean < Distance:
                    Distance = Euclidean # Update Error approximation
        """
        Distances.append(Distance)
        if Distance == 2* searchWidh * np.sqrt(2):
            countD2 += 1
    RMSE2 = 0
    for Error in Distances:
        RMSE2 += Error**2
    RMSE2 = np.sqrt(RMSE2/float(len(Distances)))
    return [RMSE1,RMSE2,countD1,countD2]
def Fitness_RMSE_approxPerpendicular(NBezierPoints,searchWidh,IndividualGenotype,TargetPoints):
    """
    -> From the genotype, get the phenotype (NBezierPoints points per Bezier curve)
    -> For every contour target point:
       -> Identify if there are Bezier points within a given XY square of widh searchWidh (e.g. 9)
       -> If so, calculate and store smallest euclidean distance found
       -> If not, set 2*square diagonal as the Error
    -> Calculate and return the Root-Mean-Squared-Approx Euclidean Error: RMSE1
    -> For every Bezier point:
       -> get an integer range of interest
       -> call target points directly in the range of the cross; calculate the smallest euclidian distance
       -> If not, set 2*square diagonal as the Error
    -> Calculate and return the Root-Mean-Squared-Approx Euclidean Error: RMSE2
    -> return RMSE1 + RMSE2
    IndividualGenotype: genotype: [ [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]], [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]],...Ncurves]
    TargetPoints: [[X,Y],[X,Y],...]
    rerturns the RMSE for Individual
    """
    Distances = [] # one for every target points; the euclidean distance to approx closest Bezier point
    # Get the Bezier points (Phenotype) from the genotype, in a list of Xs and one of Ys:
    BezierXs = [] # [X(t0),X(t1),... NBezierPoints]
    BezierYs = [] # [Y(t0),Y(t1),... NBezierPoints]
    for bezier in IndividualGenotype: # each iter: curve gene [ [X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3] ]
        Bout = CubicBezierPoints(NBezierPoints,bezier) # return [[X(t0),X(t1),...],[Y(t0),Y(t1),...]]
        BezierXs += Bout[0]
        BezierYs += Bout[1]
    # Calculate the approx error for every target point
    TargetXs = []
    TargetYs = []
    countD1 = 0 #***
    for Target in TargetPoints: # each iter: [X,Y]
        Distance = 2* searchWidh * np.sqrt(2) # Initialize as the 2*diagonal of the search square
        TX = Target[0] # X of the point we want to fit
        TY = Target[1] # Y of the point we want to fit
        TargetXs.append(TX)
        TargetYs.append(TY)
        Xrange = [TX-searchWidh/2.,TX+searchWidh/2.] # Width range of the Vertical space of the search cross
        Yrange = [TY-searchWidh/2.,TY+searchWidh/2.] # Height range of the Horizontal space of the search cross
        # Find shortest distance to all Bezier points in the Xrange of search:
        for i in range(len(BezierXs)): # Check every Bezier point
            if (BezierXs[i] >= Xrange[0] and BezierXs[i] <= Xrange[1]) and (BezierYs[i] >= Yrange[0] and BezierYs[i] <= Yrange[1]):
                # i Bezier point is within search range
                Euclidean = ( (BezierXs[i]-TX)**2 + (BezierYs[i]-TY)**2 )**(0.5)
                if Euclidean < Distance:
                    Distance = Euclidean # Update Error approximation
        Distances.append(Distance)
        if Distance == 2* searchWidh * np.sqrt(2):
            countD1 += 1
    RMSE1 = 0
    for Error in Distances:
        RMSE1 += Error**2
    RMSE1 = np.sqrt(RMSE1/float(len(Distances)))
    #################################################### Calculate error from the Bezier points
    Distances = []
    countD2 = 0 #***
    for i in range(len(BezierXs)): # for every Bezier point
        Distance = 2* searchWidh * np.sqrt(2) # Initialize as the 2*diagonal of the search square
        TXs = TargetXs[:]
        TYs = TargetYs[:]
        Bx = BezierXs[i]
        By = BezierYs[i]
        Xrange = range(int(Bx-searchWidh/2.),int(Bx+searchWidh/2.)+1)
        Yrange = range(int(By-searchWidh/2.),int(By+searchWidh/2.)+1)
        # look in the X range of the cross
        for X in Xrange:
            while X in TXs: # a relevant-in-X target point found!
                index = TXs.index(X)
                if TYs[index] in Yrange: # a relevant-in-XY target point found!
                    Euclidean = ( (TXs[index]-Bx)**2 + (TYs[index]-By)**2 )**(0.5)
                    if Euclidean < Distance:
                        Distance = Euclidean # Update Error approximation
                TXs[index] = "used" #This Target point won't be used again for this Bezier point
                TYs[index] = "used"
        """
        for X in Xrange:
            while X in TXs: # a relevant target point found!
                index = TXs.index(X)
                Euclidean = ( (TXs[index]-Bx)**2 + (TYs[index]-By)**2 )**(0.5)
                TXs[index] = "used" #This Target point won't be used again for this Bezier point
                TYs[index] = "used"
                if Euclidean < Distance:
                    Distance = Euclidean # Update Error approximation
        # look in the Y range of the cross
        for Y in Yrange:
            while Y in TYs: # a relevant target point found!
                index = TYs.index(Y)
                Euclidean = ( (TXs[index]-Bx)**2 + (TYs[index]-By)**2 )**(0.5)
                TXs[index] = "used" #This Target point won't be used again for this Bezier point
                TYs[index] = "used"
                if Euclidean < Distance:
                    Distance = Euclidean # Update Error approximation
        """
        Distances.append(Distance)
        if Distance == 2* searchWidh * np.sqrt(2):
            countD2 += 1
    RMSE2 = 0
    for Error in Distances:
        RMSE2 += Error**2
    RMSE2 = np.sqrt(RMSE2/float(len(Distances)))
    return [RMSE1,RMSE2,countD1,countD2]
def InvertFitness(FitnessList):
    """
    Returns [1/F1 , 1/F2, ...]
    """
    IFitnessList = []
    for F in FitnessList:
        if F == 0:
            F = 0.001
            print "A fitness value was: "+str(F)+", replaced by 0.001"
        IFitnessList.append(1/float(F))
    return IFitnessList
def SelectParents(FitnessList_maxGood,NParents):
    """
    ROULETTE WHEEL SELECTION:
    The higher the fitness value the better
    return list of NtotalParents parents [indexes]
    """
    iParents = []
    for p in range(NParents):
        i = RouletteWheelSelection(FitnessList_maxGood) # Returns the index of the individual selected
        iParents.append(i)
    return iParents
def SelectElites(FitnessList_maxGood,NElites):
    """
    NElites: No. of individuals with top fitness (highest) that will directly be in the next generation.
    return list of elites [indexes]
    """
    FitnessList = FitnessList_maxGood[:]
    iElites = []
    for E in range(NElites):
        topindex=FitnessList.index(max(FitnessList)) #Find the index of the top individual
        iElites.append(topindex)
        FitnessList[topindex] = -1
    return iElites
def CrossOver_offsprings_V1(ParentsGenotypes):
    """
    ParentsGenotypes: [Genotype1,Genotype2,...NParents]
    Genotype: [ [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]], [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]],...Ncurves]
    return Offsprings Genotypes: [Genotype1,Genotype2,...NParents/2]
    The crossover randombly mixes the curves from the parents
    """
    OffspringGenotypes = []
    NOffsprings = len(ParentsGenotypes)/2
    for child in range(NOffsprings): # For every Offspring
        P1 = random.choice(ParentsGenotypes) #Select Parent 1 randomly (uniform) from Parents list
        P2 = random.choice(ParentsGenotypes)
        GenePool = P1 + P2 # Curves from both parents
        shuffle(GenePool) # Mix the curves randomly
        ChildGenotype = GenePool[:len(GenePool)/2]
        OffspringGenotypes.append(ChildGenotype)
    return OffspringGenotypes
def CrossOver_offsprings(ParentsGenotypes,Dictionary_Clusters):
    """
    ParentsGenotypes: [Genotype1,Genotype2,...NParents]
    Genotype: [ [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster], [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster],...Ncurves]
    return Offsprings Genotypes: [Genotype1,Genotype2,...NParents/2]
    The crossover randombly mixes the curves from the parents
    Dictionary_Clusters: keys=cluster indexes, values=
    [Center[X,Y], [points[X,Y] assigned to cluster],corners[[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...] ]
    """
    OffspringGenotypes = []
    NOffsprings = len(ParentsGenotypes)/2
    NClusters = len(Dictionary_Clusters.keys())
    Ncurves = len(ParentsGenotypes[0]) # How many curves are in one genotype
    for child in range(NOffsprings): # For every Offspring
        ChildGenotype = []
        P1 = random.choice(ParentsGenotypes) #Select Parent 1 randomly (uniform) from Parents list
        P2 = random.choice(ParentsGenotypes)
        GenePool = P1 + P2 # Curves from both parents
        #shuffle(GenePool) # Mix the curves randomly
        # Build cluster dictionary
        ClusterSizes = [0]*NClusters
        DictioClusterGenePool = {}
        for k in range(NClusters):
            DictioClusterGenePool[k] = [] # Initialize lists of curves per cluster
        for curve in GenePool:
            DictioClusterGenePool[curve[4]].append(curve)
            ClusterSizes[curve[4]] += 1
        # Buil a child, first one curve from the gene pool for every cluster, then Roulette wheel based on cluster size
        iCluster = 0
        for c in range(Ncurves): # for every curve (gene) to be selected
            if iCluster >= NClusters:
                iCluster = RouletteWheelSelection(ClusterSizes) # More curves -> More likely
            Curve = random.choice(DictioClusterGenePool[iCluster])
            ChildGenotype.append(Curve)
            iCluster += 1
        OffspringGenotypes.append(ChildGenotype)
    return OffspringGenotypes
def MutatePopulation(Genotypes,Gauss_SD,mutate_probab):
    """
    Genotypes: [Genotype1,Genotype2,...NParents]
    Genotype: [ [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster], [[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster],...Ncurves]
    return mutated Genotypes
    # +/- 3.5SD; 99.95% are within https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    """
    Mutated = copy.deepcopy(Genotypes)
    for i in range(len(Genotypes)): #for every individual
        for j in range(len(Genotypes[i])): #for every curve
            for k in range(4): #for every point
                for l in range(2):
                    R = random.random() #Random uniform value between [0.0,1.0]
                    if R < mutate_probab: #Mutation probability was met!
                        MutateAmount = random.gauss(0, Gauss_SD) #Sample from a gaussian distribution Mean = 0 and SD
                        Mutated[i][j][k][l] += MutateAmount
    return Mutated
def GeneticAlgorithm_SketchRepresentation(NPopulation,propElites,NGenerations,mutate_probab,mutate_Gauss_SD,Ncurves,NBezierPoints,searchWidh,Dictio_Clusters):
    """
    NPopulation:No. of individuals in population
    propElites: Proportion [0,1] of top individuals that are passed on directly to the next generation
    NGenerations: How many evolving generations
    mutate_probab: how likely [0,1] is it for a mutation to happen at every piece of info in every gene after crossover; i.e., in the XY coordinates of the
                   4 control points that determine every cubic Bezier curve
    mutate_Gauss_SD: A Gauss distribution with this Standard Deviation will determine the magnitudes of mutation
    Ncurves: how many cubic Bezier curves per individual
    NBezierPoints: how many points are created per Bezier curves in the Phenotype
    searchWidh: For measuring the error, this is the width of the search "cross" with the target point in the center
    Dictionary_Clusters: keys=cluster indexes, values=[Center[X,Y], [points[X,Y] assigned to cluster],corners[[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...] ]
    -------------------------------------- This Genetic Algorithm MINIMIZES the fitness function, i.e., the RMSE deviations between target points and the Bezier curves
    Returns [[best genotype],its fitness,its generation,[fittest history],[All genotypes history],[Mean Fitness History],[Mean FitnessTargetP History],[Mean FitnessBezierP History]]
    where [All genotypes history]=[[Genotype1],...]; Genotype1=[[curve1],[curve2],...,Fitness,Generation]
    """
    # Retrieve Target Points:
    TargetPoints = []
    for k in Dictio_Clusters.keys():
        TargetPoints += Dictio_Clusters[k][1]
    # Initialize population:
    NElites = int(propElites*NPopulation) #Number of elites
    NOffsprings = NPopulation-NElites #Number of Offsprings
    NParents = NOffsprings*2 #Number of parents: 2 for every wanted child (without elites)
    Popu = []
    for p in range(NPopulation):
        Popu.append(InitializeIndividual(Ncurves,Dictio_Clusters))
    print "Population Initialized"
    #Initialize results lists:
    FittestHistory = []
    MeanFitnessHistory = []
    MeanFitnessTargetPHistory = []
    MeanFitnessBezierPHistory = []
    MeanFitnessNNHistory = []
    PopuHistory = []
    Fittest = [[],10000,10000] #Initialize result list
    for G in range(NGenerations):#For every generation
        print "-------------------------------------"
        print "Generation: "+str(G+1)
        #Evaluate Fitness:
        FitnessValues = []
        FitnessNN = []
        FitnessTargetPoints = []
        FitnessBezierPoints = []
        Count1s = []
        Count2s = []
        for genotype in Popu: #For every inidividual in the current population
            FitnessTargetP,FitnessBezierP,count1,count2 = Fitness_RMSE_approxPerpendicular(NBezierPoints,searchWidh,genotype,TargetPoints)
            Count1s.append(count1)
            Count2s.append(count2)
            FitnessTargetPoints.append(FitnessTargetP)
            FitnessBezierPoints.append(FitnessBezierP)

            # Get score from neural network
            nn_score = get_score(design, genotype)
            FitnessNN.append(nn_score)
            # Print for debugging purposes
            print('NN score', nn_score, 'RMSE score', FitnessTargetP + 1*FitnessBezierP)

            FitnessValues.append(FitnessTargetP + 1*FitnessBezierP + nn_score) # Bezier fitness weights 1X
            Individual = copy.deepcopy(genotype)
            Individual += [FitnessValues[-1],G+1] #Add Fitness and Generation at the end of Individual
            PopuHistory.append(Individual)
        MeanFitnessHistory.append(np.mean(FitnessValues))
        MeanFitnessTargetPHistory.append(np.mean(FitnessTargetP))
        MeanFitnessBezierPHistory.append(np.mean(FitnessBezierP))
        MeanFitnessNNHistory.append(np.mean(FitnessNN))

        FittestValue = min(FitnessValues)# fittest value of this generation
        FittestHistory.append(FittestValue)#Add the fittest of this generation to the fitness history
        #Update best results
        if FittestValue < Fittest[1]: #If the fittest of this generation is better than historic best:
            Fittest[0] = Popu[FitnessValues.index(FittestValue)] #Result: Look for the index of the fittest of this generation and insert its genotype
            Fittest[1] = FittestValue # Result: Insert the fitness of the leading chromosome
            Fittest[2] = (G+1) #Result: Insert the current generation number, +1 because it starts in 0
        print "-->Fittest now: "+str(FittestValue) + "; historical: "+str(Fittest[1])
        print "-->Mean Fitness: "+str(MeanFitnessHistory[-1])
        print "-->Mean Fitness(TargetP): "+str(FitnessTargetPoints[-1])+"; Mean out: "+str(np.mean(Count1s))
        print "-->Mean Fitness(BezierP): "+str(FitnessBezierPoints[-1])+"; Mean out: "+str(np.mean(Count2s))
        print "-->Mean Fitness(NN): "+str(MeanFitnessNNHistory[-1])
        #Select Parents:
        FitnessList_maxGood = InvertFitness(FitnessValues)
        Parents_indexes = SelectParents(FitnessList_maxGood,NParents)
        Parents = []
        for i in Parents_indexes:
            Parents.append(Popu[i])
        #Crossover, Offsprings:
        Offsprings = CrossOver_offsprings(Parents,Dictio_Clusters) # Dictio_Clusters
        #Mutate Offsprings:
        MutatedOffsprings = MutatePopulation(Offsprings,mutate_Gauss_SD,mutate_probab)
        Offsprings = [] # Empty this list
        #Select Elites:
        if NElites == 1: #If only one elite, don't mutate it
            Eindex = SelectElites(FitnessList_maxGood,1)[0]
            Elite = [Popu[Eindex]]
            Popu = Elite+MutatedOffsprings
        elif NElites > 1:#If several elites, clone the best and don't mutate her; mutate the other elites including the clone
            Nselect = NElites/2
            Best_indexes = SelectElites(FitnessList_maxGood,NElites-Nselect) # Find the best elite
            Bests = []
            for i in Best_indexes:
                Bests.append(Popu[i])
            Elites_indexes = SelectElites(FitnessList_maxGood,Nselect) # This will also include BestElite
            Elites = []
            for i in Elites_indexes:
                Elites.append(Popu[i])
            MutatedElites = MutatePopulation(Elites,mutate_Gauss_SD,mutate_probab) #Mutate Elites
            Elites = [] #Empty list
            Popu = Bests + MutatedElites + MutatedOffsprings
        else:
            Popu = MutatedOffsprings
        shuffle(Popu) #Reshuffle the list of chromosomes randomly
    Fittest.append(FittestHistory) #Result: Insert the fitness history
    Fittest.append(PopuHistory) #Result: Insert the chromosome History
    Fittest.append(MeanFitnessHistory)
    Fittest.append(MeanFitnessTargetPHistory)
    Fittest.append(MeanFitnessBezierPHistory)
    return Fittest
###############################################################################
###############################################################################
############################################################################### BODY OF CODE:
###############################################################################
####################################################################### INPUTS:
##### Inputs for graphs:
name = "Seed4_300dpi" # Name of the file with the input sketch
ImageOut = "Y"
TextOut = "Y"
Figs = ["*RelevantPoints","*Clusters","*Clusters_Dist","*Clusters_Corners","*TestBezier","*InitializedIndividual","*Fitness","Fittest"]
##### Inputs for characterizing the input sketch:
GrayTol = 210 # if Point RGB value < Tol; Point is considered to be relevant. From decimal 0 to 255. Black R=G=B=0; white R=G=B=255
k_Clusters = 10
maxRounds = 50 # For K-Means clustering
Tol_ProportionalDeltaDensity = 0.55 # threshold for detecting corners based on proportional density derivative in XY
Ncurves = 25 #25 # No. of cubic Bezier curves in every Genotype (Individual)
NBezierPoints = 5 #5 #how many points are created per Bezier curves in the Phenotype
searchWidh = 5 #5 #For measuring the error, this is the width of the search "cross" with the target point in the center
##### Inputs for Genetic functions:
NPopulation = 50 #50 #No. of individuals in population
propElites = 0.5 #0.5 #Proportion [0,1] of top individuals that are passed on directly to the next generation
NGenerations = 20 #20 #How many evolving generations
mutate_probab = 0.25 #0.25 #how likely [0,1] it is for a mutation to happen at every XY coordinates of the 4 control points that determine every cubic Bezier curve
mutate_Gauss_SD = 2 #2 #A Gauss distribution with this Standard Deviation will determine the magnitudes of mutation
###############################################################################

# Init design to load neural network
design = Design()

# s0 is an empty image by default
# s0 = load_image('rect.png')
# design.set_design(s0)

# set s1
s1 = load_image(name + '.png')
design.set_design(s1)


start_time = time.clock() # Start time
filename = name+".png"
SketchIn = io.imread(filename)
nRows = len(SketchIn)
nColumns = len(SketchIn[0])
print "Image: "+ filename
print "Rows (height): "+str(nRows)+"; Columns (width): "+str(nColumns)
# Detect relevant points in the input sketch
Points = [] #for every relevant pixel: [X,Y]
for i in range(nRows): #iter through rows
    for j in range(nColumns): #iter through columns in row i
        Pixel = SketchIn[i][j]
        if Pixel[0] < GrayTol or Pixel[1] < GrayTol or Pixel[2] < GrayTol: # Relevant point detected!
            X = j
            Y = nRows - i
            Points.append([X,Y])
# Assing the Relevant Points to nearest K clusters
Dictio_Clusters = KMeansClusters(Points,k_Clusters,maxRounds,0,nColumns,0,nRows)
# To every cluster key, add the potential corners
nCorners = 0
for k in Dictio_Clusters.keys(): # For every Cluster
    Corners = CornersDetector(Dictio_Clusters[k][1],Tol_ProportionalDeltaDensity)
    nCorners += len(Corners)
    Dictio_Clusters[k].append(Corners) # Add Corners: [[CornerX0,CornerY0,Fitness0],[CornerX1,CornerY1,,Fitness0],...]
print "Potential Corners: "+str(nCorners)

# Genetic Algorithm for finding a Sketch Representation with cubic Beziers by minimizing the Euclidean RMSE (approx):
GAResults=GeneticAlgorithm_SketchRepresentation(NPopulation,propElites,NGenerations,mutate_probab,mutate_Gauss_SD,Ncurves,NBezierPoints,searchWidh,Dictio_Clusters)
#Returns [[best genotype],its fitness,its generation,[fittest history],[genotypes history with Fitness,G],[Mean Fitness History],[Mean FitnessTargetP History],[Mean FitnessBezierP History]]
BestGenotype = GAResults[0]
Bestfitness = GAResults[1]
GenerationOfFittest = GAResults[2]
FittestHistory = GAResults[3]
GenotypesHistory = GAResults[4] #These are all the geneotypes throughout the evolution; nP*NG
MeanFitnessHistory = GAResults[5]
MeanFitnessTargetPHistory = GAResults[6]
MeanFitnessBezierPHistory = GAResults[7]
print "---------------- GA DONE!"
print "***FITTEST: "+str(Bestfitness)+", in generation: "+str(GenerationOfFittest)
print "Fittest 1st G: "+str(FittestHistory[0])+"; last G: "+str(FittestHistory[-1])
print "Mean Fitness 1st G: "+str(MeanFitnessHistory[0])+"; last G: "+str(MeanFitnessHistory[-1])
print "Mean Fitness(TargetP) 1st G: "+str(MeanFitnessTargetPHistory[0])+"; last G: "+str(MeanFitnessTargetPHistory[-1])
print "Mean Fitness(BezierP) 1st G: "+str(MeanFitnessBezierPHistory[0])+"; last G: "+str(MeanFitnessBezierPHistory[-1])
# Print time
print time.clock() - start_time, "seconds: Execution time"
# Text out:
TextOut_name = name+'_OutGenotypes_NP'+str(NPopulation)+'_pE'+str(propElites)+'_NG'+str(NGenerations)+'_pM'+str(mutate_probab)+'_SD'+str(mutate_Gauss_SD)+'.txt'
if TextOut == "Y":
    out_file = open(TextOut_name,'w')
    out_file.write("GrayTol,"+str(GrayTol)+",k_Clusters,"+str(k_Clusters)+",maxRounds,"+str(maxRounds)+",Tol_ProportionalDeltaDensity,"+str(Tol_ProportionalDeltaDensity))
    out_file.write(",Ncurves,"+str(Ncurves)+",NBezierPoints,"+str(NBezierPoints)+",searchWidh,"+str(searchWidh)+",NPopulation,"+str(NPopulation))
    out_file.write(",propElites,"+str(propElites)+",NGenerations,"+str(NGenerations)+",mutate_probab,"+str(mutate_probab)+",mutate_Gauss_SD,"+str(mutate_Gauss_SD)+"\n")
    out_file.write("__________________________________________________________\n")
    out_file.write("Genotype,Generation,Fitness,Ncurves where curve=[[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3],cluster]\n")
    i = 0
    for g in GenotypesHistory: #for every individual Genotype
        out_file.write(str(i)+","+str(g[-1])+","+str(g[-2]))
        for c in g[:-2]: # for every curve
            out_file.write(","+str(c[0][0])+","+str(c[0][1])+","+str(c[1][0])+","+str(c[1][1])+","+str(c[2][0])+","+str(c[2][1])+","+str(c[3][0])+","+str(c[3][1])+","+str(c[4]))
        out_file.write("\n")
        i+=1
    out_file.close()
###############################################################################
###############################################################################
###############################################################################
############################################################### GRAPHS
#In matplotlib pass color = [(0.3,0.3,0.5)]; where (r, g, b, a)
plt.close("all")
Size = 1.5 # Marker size

if "RelevantPoints" in Figs:
    fig1 = plt.figure()
    fig1.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1)
    R,G,B = 200/255.,200/255.,200/255.
    for P in Points:
        plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    if ImageOut == "Y":
        pp = PdfPages(name+'_RelevantPoints.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "Clusters" in Figs:
    fig2 = plt.figure()
    fig2.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1)
    for k in Dictio_Clusters.keys(): # For every Cluster
        R = random.uniform(0, 1)
        G = random.uniform(0, 1)
        B = random.uniform(0, 1)
        plt.plot(Dictio_Clusters[k][0][0],Dictio_Clusters[k][0][1],marker='x',markersize=10,color=(R,G,B)) # Plot cluster centers
        for P in Dictio_Clusters[k][1]: # For every point in this cluster
            plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages(name+'_PointClusters.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "(not available)Clusters_Dist" in Figs:
    fig3 = plt.figure()
    fig3.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1)
    for k in Dictio_Clusters.keys(): # For every Cluster
        R = random.uniform(0, 1)
        G = random.uniform(0, 1)
        B = random.uniform(0, 1)
        # Plot cluster centers
        plt.plot(Dictio_Clusters[k][0][0],Dictio_Clusters[k][0][1],marker='x',markersize=10,color=(R,G,B))
        for P in Dictio_Clusters[k][1]: # For every point in this cluster
            plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
        #Distribution: horizontal Distribution Bars
        for i in range(len(Dictio_Clusters[k][2])):
            plt.plot([Dictio_Clusters[k][2][i],Dictio_Clusters[k][2][i]],[Dictio_Clusters[k][0][1]+70*Dictio_Clusters[k][4][i],Dictio_Clusters[k][0][1]],color="k",linewidth=0.5)
        #Distribution: horizontal line
        plt.plot([Dictio_Clusters[k][2][0],Dictio_Clusters[k][2][-1]],[Dictio_Clusters[k][0][1],Dictio_Clusters[k][0][1]],"-k",linewidth=0.5)
        #Distribution: Vertical Distribution Bars
        for i in range(len(Dictio_Clusters[k][3])):
            plt.plot([Dictio_Clusters[k][0][0]+70*Dictio_Clusters[k][5][i],Dictio_Clusters[k][0][0]],[Dictio_Clusters[k][3][i],Dictio_Clusters[k][3][i]],color="k",linewidth=0.5)
        #Distribution: Vertical line
        plt.plot([Dictio_Clusters[k][0][0],Dictio_Clusters[k][0][0]],[Dictio_Clusters[k][3][0],Dictio_Clusters[k][3][-1]],"-k",linewidth=0.5)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages(name+'_PointClusters_Distri.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "Clusters_Corners" in Figs:
    fig4 = plt.figure()
    fig4.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1)
    for k in Dictio_Clusters.keys(): # For every Cluster
        R = random.uniform(0, 1)
        G = random.uniform(0, 1)
        B = random.uniform(0, 1)
        #plt.plot(Dictio_Clusters[k][0][0],Dictio_Clusters[k][0][1],marker='x',markersize=10,color=(R,G,B)) # Plot cluster centers
        # Plot points in cluster:
        for P in Dictio_Clusters[k][1]: # For every point in this cluster
            plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
        # Plot corners in cluster:
        Corners = Dictio_Clusters[k][2]
        for c in Corners:
            size = 5*(c[2]/Tol_ProportionalDeltaDensity) # larger corner fitness -> larger marker
            plt.plot([c[0]],[c[1]],marker='x',markersize=size,color=(R,G,B))
        '''
        #Distribution: horizontal Distribution Bars
        for i in range(len(Dictio_Clusters[k][2])):
            plt.plot([Dictio_Clusters[k][2][i],Dictio_Clusters[k][2][i]],[Dictio_Clusters[k][0][1]+70*Dictio_Clusters[k][4][i],Dictio_Clusters[k][0][1]],color="k",linewidth=0.5)
        #Distribution: horizontal line
        plt.plot([Dictio_Clusters[k][2][0],Dictio_Clusters[k][2][-1]],[Dictio_Clusters[k][0][1],Dictio_Clusters[k][0][1]],"-k",linewidth=0.5)
        #Distribution: Vertical Distribution Bars
        for i in range(len(Dictio_Clusters[k][3])):
            plt.plot([Dictio_Clusters[k][0][0]+70*Dictio_Clusters[k][5][i],Dictio_Clusters[k][0][0]],[Dictio_Clusters[k][3][i],Dictio_Clusters[k][3][i]],color="k",linewidth=0.5)
        #Distribution: Vertical line
        plt.plot([Dictio_Clusters[k][0][0],Dictio_Clusters[k][0][0]],[Dictio_Clusters[k][3][0],Dictio_Clusters[k][3][-1]],"-k",linewidth=0.5)
        '''
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages(name+'_PointClusters_Corners.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "TestBezier" in Figs:
    # Bezier cubic curves:
    Point0 = [0,4]
    Point1 = [1.3,8]
    Point2 = [7.5,6.5]
    Point3 = [6,10]
    BezierPoints = CubicBezierPoints(NBezierPoints,[Point0,Point1,Point2,Point3])
    fig5 = plt.figure()
    #plt.xlim(-nColumns*0.1,nColumns*1.1)
    #plt.ylim(-nRows*0.1,nRows*1.1)
    plt.plot(BezierPoints[0],BezierPoints[1],"-c",linewidth = 2)
    plt.plot(BezierPoints[0],BezierPoints[1],marker='o',markersize = 3,markeredgewidth=0,linewidth=0,color = "k")
    Color = 200/255.
    plt.plot([Point0[0],Point1[0]],[Point0[1],Point1[1]],"--",linewidth = 1.5,color=(Color,Color,Color))
    plt.plot([Point1[0],Point2[0]],[Point1[1],Point2[1]],"--",linewidth = 1.5,color=(Color,Color,Color))
    plt.plot([Point2[0],Point3[0]],[Point2[1],Point3[1]],"--",linewidth = 1.5,color=(Color,Color,Color))
    marker = 10
    plt.plot([Point0[0]], [Point0[1]],marker='x', markersize=marker,markeredgewidth=1,color = "r")
    plt.plot([Point1[0]], [Point1[1]],marker='x', markersize=marker,markeredgewidth=1,color = "r")
    plt.plot([Point2[0]], [Point2[1]],marker='x', markersize=marker,markeredgewidth=1,color = "r")
    plt.plot([Point3[0]], [Point3[1]],marker='x', markersize=marker,markeredgewidth=1,color = "r")
    #plt.grid()
    if ImageOut == "Y":
        pp = PdfPages('CubicBezierTest.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "InitializedIndividual" in Figs:
    #Initialize one Individual:
    Genotype1 = InitializeIndividual(Ncurves,Dictio_Clusters)
    fig6 = plt.figure()
    fig6.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1)
    R,G,B = 200/255.,200/255.,200/255.
    for P in Points:
        plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
    marker = 5
    for curve in Genotype1:
        XsBezier, YsBezier= CubicBezierPoints(NBezierPoints,curve)
        R = random.uniform(0, 1)
        G = random.uniform(0, 1)
        B = random.uniform(0, 1)
        #Plot Bezier curve:
        plt.plot(XsBezier,YsBezier,linestyle='-',linewidth=1,color=(R,G,B))
        #Plot the control Points:
        for P in curve[:-1]:
            plt.plot([P[0]], [P[1]],marker='x', markersize=marker,markeredgewidth=0.5,color=(R,G,B))
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages(name+'_InitializedIndividual.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "Fitness" in Figs:
    fig7 = plt.figure()
    plt.xlim(1,len(FittestHistory))
    plt.ylim(0,max(MeanFitnessHistory)*1.1)
    plt.plot(range(1,len(FittestHistory)+1),MeanFitnessHistory,"-k",linewidth = 2,label="Mean Fitness")
    plt.plot(range(1,len(FittestHistory)+1),MeanFitnessTargetPHistory,"--c",linewidth = 2,label="Mean Fitness(Target Points)")
    plt.plot(range(1,len(FittestHistory)+1),MeanFitnessBezierPHistory,"--b",linewidth = 2,label="Mean Fitness(Bezier Points)")
    plt.plot(range(1,len(FittestHistory)+1),FittestHistory,"-r",linewidth = 2,label="Fittest")
    plt.plot(range(1,len(FittestHistory)+1),FittestHistory,marker='o',markersize = 3,markeredgewidth=0,linewidth=0,color = "k")
    marker = 10
    plt.plot([GenerationOfFittest], [Bestfitness],marker='x', markersize=marker,markeredgewidth=1,color = "r")
    plt.grid()
    plt.legend(loc='best', fancybox=True, framealpha=0.8,fontsize=10)
    plt.xlabel('Generations',fontsize=14)
    plt.ylabel('Fitness',fontsize=14)
    if ImageOut == "Y":
        pp = PdfPages('GA_FitnessHistory_NP'+str(NPopulation)+'_pE'+str(propElites)+'_NG'+str(NGenerations)+'_pM'+str(mutate_probab)+'_SD'+str(mutate_Gauss_SD)+'.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "Fittest" in Figs:
    fig8 = plt.figure()
    fig8.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1)
    #Plot Target Points:
    R,G,B = 240/255.,240/255.,240/255.
    for P in Points:
        plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
    #Plot Bezier curves:
    BezierColor = (100/255.,100/255.,100/255.,0.6) # {R,G,B,a}
    for curve in BestGenotype:
        XsBezier, YsBezier= CubicBezierPoints(50,curve)
        plt.plot(XsBezier,YsBezier,linestyle='-',linewidth=0.8,color=BezierColor)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages(name+'_Fittest_NP'+str(NPopulation)+'_pE'+str(propElites)+'_NG'+str(NGenerations)+'_pM'+str(mutate_probab)+'_SD'+str(mutate_Gauss_SD)+'.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()


"""IDEAS:
-> You can easily estimate the tanget (Y') and compare it to the reference structure to measure divergence
"""
