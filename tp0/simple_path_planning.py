# %do_load import
import pinocchio as pin
import time
import numpy as np
from numpy.linalg import inv,norm,pinv,svd,eig
from scipy.optimize import fmin_bfgs,fmin_slsqp
from utils.load_ur5_with_obstacles import load_ur5_with_obstacles,Target
import matplotlib.pylab as plt
# %end_load
plt.ion()  # matplotlib with interactive setting

# %do_load robot
robot = load_ur5_with_obstacles()
# %end_load

# %do_load viewer
#robot.initViewer(loadModel=True)
#viz = robot.viz
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
viz.display(robot.q0)
# %end_load

# %do_load target
target = Target(robot.viz,position = np.array([.5,.5]))
# %end_load

################################################################################
################################################################################
################################################################################

# %do_load q2_to_q6
def q2_to_q6(q2):
     '''
     Transform a vector 2 into a vector 6, corresponding to locking 4 joints of the 6-dof arm. 
     '''
     q6 = np.zeros(6)
     q6.flat[[1,2]] = q2
     return q6

def q6_to_q2(q6):
     return q6[1:3]
# %end_load

# %do_load endef
def endef(q):
     '''Return the 2d position of the end effector.'''
     pin.forwardKinematics(robot.model,robot.data,q)
     return robot.data.oMi[-1].translation[[0,2]]
# %end_load

# %do_load  dist
def dist(q):
     '''Return the distance between the end effector end the target (2d).'''
     return norm(endef(q)-target.position)
# %end_load

# %do_load coll
def coll(q):
     '''Return true if in collision, false otherwise.'''
     pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
     return pin.computeCollisions(robot.collision_model,robot.collision_data,False)
# %end_load

# %do_load qrand
def qrand(check=False):
    '''
    Return a random configuration. If check is True, this
    configuration is not is collision
    '''
    while True:
        q = q2_to_q6(np.random.rand(2)*6-3)  # sample between -3 and +3.
        if not check or not coll(q): return q
# %end_load

# %do_load colldist
def collisionDistance(q):
     '''Return the minimal distance between robot and environment. '''
     threshold = 1e-2
     pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
     if pin.computeCollisions(robot.collision_model,robot.collision_data,False): return -threshold
     idx = pin.computeDistances(robot.collision_model,robot.collision_data)
     return pin.computeDistance(robot.collision_model,robot.collision_data,idx).min_distance - threshold
# %end_load
################################################################################
################################################################################
################################################################################

# %do_load qrand_target
# Sample a random free configuration where dist is small enough.
def qrandTarget(threshold=5e-2, display=False):
     while True:
          q = qrand()
          if display:
               viz.display(q)
               time.sleep(1e-3)
          if not coll(q) and dist(q)<threshold:
               return q
viz.display(qrandTarget())
# %end_load

################################################################################
################################################################################
################################################################################

# %do_load random_descent
# Random descent: crawling from one free configuration to the target with random
# steps.
def randomDescent():
     q = qrand(check=True)
     for i in range(100):
          dq = qrand()*.1                           # Choose a random step ...
          qtry = q+dq                               # ... apply
          if dist(q)>dist(q+dq) and not coll(q+dq): # If distance decrease without collision ...
               q = q+dq                             # ... keep the step
               viz.display(q)                       # ... display it
               time.sleep(5e-3)                     # ... and sleep for a short while
# %end_load
# randomDescent()

################################################################################
################################################################################
################################################################################

# %do_load sample
def sampleSpace(nbSamples=500):
     '''
     Sample nbSamples configurations and store them in two lists depending
     if the configuration is in free space (hfree) or in collision (hcol), along
     with the distance to the target and the distance to the obstacles.
     '''
     hcol = []
     hfree = []
     for i in range(nbSamples):
          q = qrand(False)
          if not coll(q):
               hfree.append( list(q6_to_q2(q).flat) + [ dist(q), collisionDistance(q) ])
          else:
               hcol.append(  list(q6_to_q2(q).flat) + [ dist(q), 1e-2 ])
     return hcol,hfree

def plotConfigurationSpace(hcol,hfree):
     '''
     Plot 2 "scatter" plots: the first one plot the distance to the target for 
     each configuration, the second plots the distance to the obstacles (axis q1,q2, 
     distance in the color space).
     '''
     htotal = hcol + hfree
     h=np.array(htotal)
     plt.subplot(2,1,1)
     plt.scatter(h[:,0],h[:,1],c=h[:,2],s=20,lw=0)
     plt.title("Distance to the target")
     plt.colorbar()
     plt.subplot(2,1,2)
     plt.scatter(h[:,0],h[:,1],c=h[:,3],s=20,lw=0)
     plt.title("Distance to the obstacles")
     plt.colorbar()

hcol,hfree = sampleSpace(100)
plotConfigurationSpace(hcol,hfree)
# %end_load

################################################################################
################################################################################
################################################################################

# %do_load optim
def cost(q):
     '''
     Cost function: distance to the target
     '''
     return dist(q2_to_q6(q))
     
def constraint(q):
     '''
     Constraint function: distance to the obstacle should be positive.
     '''
     return collisionDistance(q2_to_q6(q))
     
def callback(q):
     '''
     At each optimization step, display the robot configuration in gepetto-viewer.
     '''
     viz.display(q2_to_q6(q))
     time.sleep(.01)

def optimize():
     '''
     Optimize from an initial random configuration to discover a collision-free
     configuration as close as possible to the target. 
     '''
     return fmin_slsqp(x0=q6_to_q2(qrand(check=True)),
                       func=cost,
                       f_ieqcons=constraint,callback=callback,
                       full_output=1)
optimize()
# %end_load

# %do_load useit
while True:
    res=optimize()
    q=q2_to_q6(res[0])
    viz.display(q)
    if res[4]=='Optimization terminated successfully' and res[1]<1e-6:
        print('Finally successful!')
        break
    print("Failed ... let's try again! ")
# %end_load
