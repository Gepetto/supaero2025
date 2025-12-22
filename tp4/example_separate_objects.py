"""
Integrative example using the collision model.
Load a complex scene (pills box), then iteratively applies an ad-hoc
repulsive field to each pair of objects that are colliding. The script
then converges to a collision free scene, with distance margin.
The movement of the scene is displayed in meshcat in real-time.
The convergence of the distances of each collision pair with negative
(colliding) distances is finally plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from create_rigid_contact_models_for_hppfcl import (
    createContactModelsFromCollisions,
    createContactModelsFromDistances,
)
from display_collision_patches import preallocateVisualObjects, updateVisualObjects
from scenes import buildScenePillsBox

from supaero2025.meshcat_viewer_wrapper import MeshcatVisualizer
from tp4.compatibility import P3X

# Create scene with multiple objects
model, geom_model = buildScenePillsBox(nobj=30)

# Create the corresponding data to the models
data = model.createData()
geom_data = geom_model.createData()

# Chose a radom configuration.
q = pin.randomConfiguration(model)
# If you want to make objects close to each other (to have multiple
# collisions), uncomment the following line
# for i in range(3): q[i::7]/=4

# Visualize ...
# Add contact patches to the visual model (this is slow)
visual_model = geom_model.copy()
preallocateVisualObjects(visual_model, 100)
viz = MeshcatVisualizer(
    model=model, collision_model=geom_model, visual_model=visual_model
)
updateVisualObjects(model, data, [], [], visual_model, viz)
viz.display(q)

# ### MAIN LOOP
# ### MAIN LOOP
# ### MAIN LOOP

# Compute a collision free configuration by pushing away the colliding bodies
# (false physics)

# %jupyter_snippet hyperparams
# HYPER PARAMETERS OF THE PUSH STRATEGY
PUSH_FACTOR = 0.1
EPSILON = 1e-1
NB_ITER = 100
# Compute the contact information based on distances or collisions?
USE_DISTANCE = True
# %end_jupyter_snippet
# With P2X, only USE_DISTANCE=True is reasonible
assert USE_DISTANCE or P3X

# %jupyter_snippet loop
# Set minimal distance to be EPSILON
# (can be zero, but the rendering is less clear).
for r in geom_data.collisionRequests:
    r.security_margin = EPSILON

# Keep distance history for active pairs (indexed by contact name)
h_dist = {}

# Iteratively push the colliding pairs ...
for i in range(NB_ITER):
    # We will compute a change of configuration dq.
    # 0 if no active pair.
    dq = np.zeros(model.nv)

    # Compute the collision at current configuration.
    if USE_DISTANCE:
        pin.computeDistances(model, data, geom_model, geom_data, q)
    else:
        pin.computeCollisions(model, data, geom_model, geom_data, q)

    # From hppfcl contact information, build a pin.RigidContactModel
    if USE_DISTANCE:
        contact_models = createContactModelsFromDistances(
            model, data, geom_model, geom_data, EPSILON
        )
    else:
        contact_models = createContactModelsFromCollisions(
            model, data, geom_model, geom_data
        )
    contact_datas = [cm.createData() for cm in contact_models]

    # For each detected contact ...
    for cmodel, cdata in zip(contact_models, contact_datas):
        # Recover contact information
        jid1 = cmodel.joint1_id
        j1Mc1 = cmodel.joint1_placement
        jid2 = cmodel.joint2_id
        j2Mc2 = cmodel.joint2_placement

        # Compute signed distance
        oMc1 = cdata.oMc1 = data.oMi[jid1] * j1Mc1
        oMc2 = cdata.oMc2 = data.oMi[jid2] * j2Mc2
        dist = oMc1.actInv(oMc2.translation)[2] - EPSILON  # signed distance

        # Decide push velocity at contact point, in contact frames
        # We apply the same velocity proportional to the inverse of the signed
        # distance (exponential repel until EPSILON distance is reached)
        c_v = PUSH_FACTOR * pin.Motion(np.array([0, 0, dist]), np.zeros(3))

        # The velocities in joint space are expressed as spatial velocity
        # at the center of the respective joints. Displace c_v from F_c to F_j1/F_j2
        # Displacement for body 1
        j1_v = j1Mc1.act(c_v)
        dq[model.idx_vs[jid1] : model.idx_vs[jid1] + 6] += j1_v.vector
        # Displacement for body 2
        j2_v = j2Mc2.act(c_v)
        dq[model.idx_vs[jid2] : model.idx_vs[jid2] + 6] -= j2_v.vector

        # Log the distance in h_dist for future plot
        if cmodel.name not in h_dist:
            h_dist[cmodel.name] = np.zeros(NB_ITER)
        h_dist[cmodel.name][i] = dist

    # Finally, modify the current config q with the push dq
    q = pin.integrate(model, q, dq)

    print(
        f"Iteration #{i}/{NB_ITER}: {len(contact_models)} active collision "
        + f"pairs (given margin {EPSILON})"
    )

    # Display the current configuration
    if i % 10 == 0:
        # Meshcat is slow to display the patches, display once in a while
        updateVisualObjects(
            model, data, contact_models, contact_datas, visual_model, viz
        )
        viz.display(q)
# %end_jupyter_snippet

# %jupyter_snippet plot
# Plot the distances
for k, v in h_dist.items():
    h = plt.plot(v, label=k)
plt.legend()
# %end_jupyter_snippet

print("Press plt.show() to display the plots.")
