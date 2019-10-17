import numpy as np
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from skimage import io
import random
#from random import shuffle
#import copy 
#import time
############################################################## FUNCTIONS
def CubicBezierPoints(n,Points):
    """
    The curve is defined from the 4 points, where e.g. Points=[ [X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3] ].
    n points are returned being equidistant in parameter t
    return [[X(t0),X(t1),...],[Y(t0),Y(t1),...]]
    https://www.desmos.com/calculator/cahqdxeshd
    """
    tlist = np.linspace(0,1,n)
    Xlist = []
    Ylist = []
    Point0,Point1,Point2,Point3 = Points
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
def SelectRoulette(FitnessList_maxGood,Nselect,Genotypes):
    """ 
    ROULETTE WHEEL SELECTION:
    The higher the fitness value the better
    return list of Nselect [Genotypes] with no repetitions
    """ 
    FitnessList = FitnessList_maxGood[:]
    Selected = []
    while len(Selected)<Nselect:
        Index=RouletteWheelSelection(FitnessList) # Returns the index of the individual selected
        if Genotypes[Index] not in Selected:
            Selected.append(Genotypes[Index] )
        FitnessList[Index] = 0
    return Selected    
def SelectElites(FitnessList_maxGood,NElites,Genotypes):
    """
    NElites: No. of individuals with top fitness (highest) that will directly be in the next generation.
    return list of [Elite Genotypes] with no repetitions
    """
    FitnessList = FitnessList_maxGood[:]
    Elites = []
    while len(Elites)<NElites:
        topindex=FitnessList.index(max(FitnessList)) #Find the index of the top individual
        if Genotypes[topindex] not in Elites:
            Elites.append(Genotypes[topindex] )
        FitnessList[topindex] = -1
    return Elites
###############################################################################
###############################################################################
############################################################################### BODY OF CODE:
############################################################################### 
####################################################################### INPUTS:
Figs = ["*Individual","*Generation","Selected","Fusion"]
ImageOut = "Y"
FileIn = "Seed3_300dpi_OutGenotypes_NP50_pE0.5_NG20_pM0.25_SD2.txt" # Input Textfile with the Genotypes
name = "Seed3_300dpi" # Name of the file with the input sketch
## Of interest:
individual = 1# For image "Individual", which one is of interest?
generation = 20 # For image "Generation", which one is of interest? Starts from 1
## Fusion:
NFuse = 10 # No. of genotypes to be fused together
Selection = "roulette" #"top"/"roulette"/"generation" # top: the top NFuse; roulette: Roulette Wheel selection based on fitness
###############################################################################
######################################################### Read Genotypes file:
"""
Genotypes=[Genotype1,...]; Genotype=[curve1,...,Fitness,G]; curve=[[X0,Y0],[X1,Y1],[X2,Y2],[X3,Y3]]
"""
read_file= open(FileIn,"r")
rows_file = read_file.read().split('\n')
Ncurves = int(rows_file[0].split(',')[9])
NPopulation = int(rows_file[0].split(',')[15])
NGenerations =int(rows_file[0].split(',')[19])
GrayTol = int(rows_file[0].split(',')[1])
rows_file = rows_file[3:-1]
Genotypes = []
for G in rows_file: # for every genotype
    Individual = []
    infoCurves = G.split(',')
    for c in range(Ncurves): # for every curve
        Curve = [[float(infoCurves[3+9*c]),float(infoCurves[4+9*c])],[float(infoCurves[5+9*c]),float(infoCurves[6+9*c])],[float(infoCurves[7+9*c]),float(infoCurves[8+9*c])],[float(infoCurves[9+9*c]),float(infoCurves[10+9*c])]]
        Individual.append(Curve)
    Individual.append(float(infoCurves[2])) # Add Fitness
    Individual.append(float(infoCurves[1])) # Add Generation
    Genotypes.append(Individual)
read_file.close()
############################################################## Relevant Points from Input Sketch
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
############################################################## Fusion:
FitnessList = []
for G in Genotypes:
    FitnessList.append(G[-2])
FitnessList_maxGood = InvertFitness(FitnessList)
if Selection == "top":
    FuseGenotypes = SelectElites(FitnessList_maxGood,NFuse,Genotypes)
elif Selection == "roulette":
    FuseGenotypes = SelectRoulette(FitnessList_maxGood,NFuse,Genotypes)
elif Selection == "generation":
    FuseGenotypes = Genotypes[NPopulation*(generation-1):NPopulation+NPopulation*(generation-1)] 
###############################################################################
###################################################################### GRAPHS:
#In matplotlib pass color = [(0.3,0.3,0.5)]; where (r, g, b, a) 
plt.close("all")
Size = 1.5 # Marker size
if "Individual" in Figs:
    fig1 = plt.figure()
    fig1.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1) 
    #Plot Target Points:
    R,G,B = 240/255.,240/255.,240/255.
    for P in Points:
        plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
    #Plot Bezier curves:
    BezierColor = (100/255.,100/255.,100/255.,0.6) # {R,G,B,a}
    for curve in Genotypes[individual][:-2]:
        XsBezier, YsBezier= CubicBezierPoints(50,curve)     
        plt.plot(XsBezier,YsBezier,linestyle='-',linewidth=0.8,color=BezierColor)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages("Sketch_Fusion_"+name+'_Individual'+str(individual)+'.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()
if "Generation" in Figs:
    # Find the individual genotypes:
    GGenotypes = Genotypes[NPopulation*(generation-1):NPopulation+NPopulation*(generation-1)]   
    fig2 = plt.figure()
    fig2.set_size_inches(1*(nColumns/float(nRows))*(10),1*(5),forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.01*(10),nColumns*1.01*(10))
    plt.ylim(-nRows*0.01*(5),nRows*1.01*(5)) 
    R,G,B = 230/255.,230/255.,230/255. # Color of the Points of interest 
    BezierColor = (90/255.,90/255.,90/255.,0.6) # {R,G,B,a}
    PointsGenX = []
    PointsGenY = []
    i = 0
    for h in range(10): # For every column (10) of sketches 
        for v in range(5): # For every row (5) of sketches
            #Plot the Tartget relevant points for the whole generation
            for P in Points:
                PointsGenX.append(P[0]+h*nColumns)
                PointsGenY.append(P[1]+v*nRows)
                #plt.plot([P[0]+h*nColumns], [P[1]+v*nRows],marker='o', markersize=Size/5.,markeredgewidth=0,color = (R,G,B))
            for curve in GGenotypes[i][:-2]:
                XsBezier, YsBezier= CubicBezierPoints(10,curve) 
                AddX = [h*nColumns] * len(XsBezier)
                AddY = [v*nRows] * len(XsBezier)
                XsBezier = np.add(XsBezier,AddX)
                YsBezier = np.add(YsBezier,AddY)
                plt.plot(XsBezier,YsBezier,linestyle='-',linewidth=0.2,color=BezierColor)
            i += 1         
    plt.plot(PointsGenX, PointsGenY,marker='o', markersize=Size/5.,markeredgewidth=0,linewidth=0,color = (R,G,B))
    #Plot Bezier curves:
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        pp = PdfPages("Sketch_Fusion_"+name+'_Generation'+str(generation)+'.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()
if "Selected" in Figs:
    fig3 = plt.figure()
    fig3.set_size_inches(2*(nColumns/float(nRows))*(5),2*(2),forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.01*(5),nColumns*1.01*(5))
    plt.ylim(-nRows*0.01*(2),nRows*1.01*(2)) 
    R,G,B = 230/255.,230/255.,230/255. # Color of the Points of interest 
    BezierColor = (90/255.,90/255.,90/255.,0.6) # {R,G,B,a}
    PointsGenX = []
    PointsGenY = []
    i = 0
    for h in range(5): # For every column (10) of sketches 
        for v in range(2): # For every row (5) of sketches
            #Plot the Tartget relevant points for the whole generation
            for P in Points:
                PointsGenX.append(P[0]+h*nColumns)
                PointsGenY.append(P[1]+v*nRows)
                #plt.plot([P[0]+h*nColumns], [P[1]+v*nRows],marker='o', markersize=Size/5.,markeredgewidth=0,color = (R,G,B))
            i += 1         
    plt.plot(PointsGenX, PointsGenY,marker='o', markersize=Size/2.,markeredgewidth=0,linewidth=0,color = (R,G,B))
    #Plot Bezier curves:
    i = 0
    for h in range(5): # For every column (10) of sketches 
        for v in range(2): # For every row (5) of sketches
            for curve in FuseGenotypes[i][:-2]:
                XsBezier, YsBezier= CubicBezierPoints(10,curve) 
                AddX = [h*nColumns] * len(XsBezier)
                AddY = [v*nRows] * len(XsBezier)
                XsBezier = np.add(XsBezier,AddX)
                YsBezier = np.add(YsBezier,AddY)
                plt.plot(XsBezier,YsBezier,linestyle='-',linewidth=0.4,color=BezierColor)
            i += 1 
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        if Selection == "generation":
            pp = PdfPages("Sketch_Fusion_"+name+'_Selected'+str(NFuse)+'_'+str(Selection)+str(generation)+'.pdf')
        else:
            pp = PdfPages("Sketch_Fusion_"+name+'_Selected'+str(NFuse)+'_'+str(Selection)+'.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()

if "Fusion" in Figs:
    fig4 = plt.figure()
    fig4.set_size_inches(5*(nColumns/float(nRows)),5,forward=True)
    # Expand the figure proportionally
    plt.xlim(-nColumns*0.1,nColumns*1.1)
    plt.ylim(-nRows*0.1,nRows*1.1) 
    #Plot Target Points:
    R,G,B = 240/255.,240/255.,240/255. # Points colors
    BezierColor = (90/255.,90/255.,90/255.,0.3) # {R,G,B,a}  # Curves colors
    for P in Points:
        plt.plot([P[0]], [P[1]],marker='o', markersize=Size,markeredgewidth=0,color = (R,G,B))
    #Plot Bezier curves:
    for G in FuseGenotypes: # For every individual Genotype Selected
        for curve in G[:-2]:
            XsBezier, YsBezier= CubicBezierPoints(50,curve) 
            plt.plot(XsBezier,YsBezier,linestyle='-',linewidth=0.3,color=BezierColor)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    if ImageOut == "Y":
        if Selection == "generation":
            pp = PdfPages("Sketch_Fusion_"+name+'_Fusion_N'+str(NFuse)+'_'+str(Selection)+str(generation)+'.pdf')
        else:
            pp = PdfPages("Sketch_Fusion_"+name+'_Fusion_N'+str(NFuse)+'_'+str(Selection)+'.pdf')
        plt.savefig(pp, format='pdf')
        pp.close()


