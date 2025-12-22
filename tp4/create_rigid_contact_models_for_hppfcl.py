"""
Define the functions to create a set of pin.RigidConstraintModels from
the results of the collision and distance functions.
"""

import numpy as np
import pinocchio as pin

# Reference vector to decide how the contact frames must be selected:
# -> with the normal to the contact along the z direction.
CONTACT_TEMPLATE_NAME = "collision_{pairId}_{contactId}"


def _createOneConstraint(
    model, data, geom_model, geom_data, pairId, OC1, OC2, normal, contactId=0
):
    """
    Atomic function to define a single 3D contact model from a collision or a distance
    point.

    From the collision pair, get the joints ID and placements.
    From the collision points and normal, compute the contact placements.

    Finally, the contact placements in joint frames are computed to create the
    RigidConstraintModel which is returned.

    <contactId> is for naming only, it should be zero when this function is called using
    the distance function (a single witness pair per body pair in that case).
    When called from collision function, multiple witness pairs can occur per pair of
    collision body, in that case this field is used to make the name of each witness
    different.
    """
    # %jupyter_snippet frames
    pair = geom_model.collisionPairs[pairId]
    gid1, gid2 = pair.first, pair.second
    g1 = geom_model.geometryObjects[gid1]
    g2 = geom_model.geometryObjects[gid2]
    jid1 = g1.parentJoint
    jid2 = g2.parentJoint
    oMj1 = data.oMi[jid1]
    oMj2 = data.oMi[jid2]

    # Compute translation and rotation of the contact placements
    # If dist=0, both placements are identical (and should be somehow close
    # when dist is reasonibly small).
    quat = pin.Quaternion.FromTwoVectors(
        pin.ZAxis, normal
    )  # orientation of the contact frame wrt world
    assert np.isclose(quat.norm(), 1)
    oMc1 = pin.SE3(quat.matrix(), OC1)  # Placement of first contact frame in world
    oMc2 = pin.SE3(quat.matrix(), OC2)  # Placement of second contact frame in world
    # %end_jupyter_snippet

    # Finally, define the contact model for this point with Pinocchio structure
    # %jupyter_snippet model
    contact_model = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        model,
        jid1,
        oMj1.inverse() * oMc1,
        jid2,
        oMj2.inverse() * oMc2,
        pin.LOCAL,
    )
    # %end_jupyter_snippet
    contact_model.name = CONTACT_TEMPLATE_NAME.format(
        pairId=pairId, contactId=contactId
    )
    return contact_model


def createContactModelsFromCollisions(model, data, geom_model, geom_data):
    """
    Create the contact models for each active collision in geom_data.
    It is supposed computeCollisions has already been called.
    For each collision, define a RigidConstraintModel with 2 reference placements:
    - one for each collision points (hopefully, they are close to each other)
    - centered at the witness point, and oriented with Z in the normal direction.
    Return a list of RigidConstraintModel.
    """

    contact_models = []
    for collId, r in enumerate(geom_data.collisionResults):
        if r.numContacts() > 0:
            for c in r.getContacts():
                OC1 = c.getNearestPoint1()  # Position of first contact point in world
                OC2 = c.getNearestPoint2()  # Position of second contact point

                # in world In some simu solver, it might be prefered to have
                # the contact point in between the collisions.
                # OC1=OC2=(OC1+OC2)/2 In 2x, this is the default behavior
                # (p1=p2=pos)

                cm = _createOneConstraint(
                    model, data, geom_model, geom_data, collId, OC1, OC2, c.normal
                )

                contact_models.append(cm)

    return contact_models


def createContactModelsFromDistances(model, data, geom_model, geom_data, threshold):
    """
    Create the contact models for each distance in geom_data whose min_distance is below
    <threshold>.
    It is supposed computeDistances has already been called.
    For each collision, define a RigidConstraintModel with 2 reference placements:
    - one for each witness points.
    - centered at the witness point, and oriented with Z in the normal direction.
    Return a list of RigidConstraintModel.
    """

    contact_models = []
    for collId, r in enumerate(geom_data.distanceResults):
        if r.min_distance > threshold:
            continue

        OC1 = r.getNearestPoint1()  # Position of first contact point in world
        OC2 = r.getNearestPoint2()  # Position of second contact point in world

        cm = _createOneConstraint(
            model, data, geom_model, geom_data, collId, OC1, OC2, r.normal
        )
        contact_models.append(cm)

    return contact_models


if __name__ == "__main__":
    from supaero2025.meshcat_viewer_wrapper import MeshcatVisualizer
    from tp4.scenes import buildSceneThreeBodies

    model, geom_model = buildSceneThreeBodies()

    data = model.createData()
    geom_data = geom_model.createData()

    # Start meshcat
    viz = MeshcatVisualizer(
        model=model, collision_model=geom_model, visual_model=geom_model
    )

    # Force the collision margin to a huge value.
    for r in geom_data.collisionRequests:
        r.security_margin = 10

    q = pin.randomConfiguration(model)
    q[2::7] += 1  # above the floor
    viz.display(q)

    # %jupyter_snippet example
    pin.computeCollisions(model, data, geom_model, geom_data, q, False)
    contact_models = createContactModelsFromCollisions(
        model, data, geom_model, geom_data
    )
    contact_datas = [cm.createData() for cm in contact_models]

    pin.computeDistances(model, data, geom_model, geom_data, q)
    contact_models = createContactModelsFromDistances(
        model, data, geom_model, geom_data, threshold=10
    )  # threshold in meter
    contact_datas = [cm.createData() for cm in contact_models]
    # %end_jupyter_snippet

    from display_witness import DisplayCollisionWitnessesInMeshcat

    wdisp = DisplayCollisionWitnessesInMeshcat(viz)
    pin.computeDistances(model, data, geom_model, geom_data, q)
    wdisp.displayDistances(geom_data)

    assert len(contact_models) == 3
