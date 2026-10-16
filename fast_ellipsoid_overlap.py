"""Conservative fast paths plus certified early-out for ellipsoid contacts.

Intersection rule and tolerance match cluster_gaussians_twice. No neighbor
truncation or approximate overlap is introduced. Axis factors avoid forming
ill-conditioned covariance matrices for thin Gaussians.
"""
import numpy as np


def ellipsoid_contacts_fast(points, factors, whitening, inner, outer, i, js,
                            extents=None, diagnostics=None):
    if extents is None:
        extents = np.linalg.norm(factors, axis=2)
    delta = points[js]-points[i]
    squared = np.einsum('bi,bi->b',delta,delta)
    accepted = squared <= (inner[i]+inner[js])**2
    possible = (~accepted) & (squared <= (outer[i]+outer[js])**2)
    possible &= np.all(np.abs(delta) <= extents[i]+extents[js],axis=1)
    positions = np.flatnonzero(possible)
    if not len(positions):
        return accepted
    ids = js[positions]
    displacements = delta[positions]
    # A shared point on the segment joining centers proves intersection.
    # This uses radial extents (inverse quadratic form), not projections.
    ai = np.linalg.norm(displacements @ whitening[i].T,axis=1)
    aj = np.linalg.norm(np.einsum('bij,bj->bi',whitening[ids],displacements),axis=1)
    witness = (1/np.maximum(ai,np.finfo(float).tiny) +
               1/np.maximum(aj,np.finfo(float).tiny)) >= 1+1e-12
    accepted[positions[witness]] = True
    if diagnostics is not None:
        diagnostics['radial_witness_accepts'] += int(witness.sum())
    positions = positions[~witness]
    if not len(positions):
        return accepted
    ids = js[positions]
    displacements = delta[positions]
    distance = np.sqrt(squared[positions])
    direction = displacements/distance[:,None]
    support_i = np.linalg.norm(direction @ factors[i],axis=1)
    support_j = np.linalg.norm(np.einsum('bji,bj->bi',factors[ids],direction),axis=1)
    separated = distance > (support_i+support_j)*(1+1e-8)
    if diagnostics is not None:
        diagnostics['separating_axis_rejects'] += int(separated.sum())
    positions = positions[~separated]
    if not len(positions):
        return accepted
    transform = whitening[i]
    eigenvectors, singular_values, _ = np.linalg.svd(transform @ factors[js[positions]])
    eigenvalues = singular_values**2
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues<=0):
        raise ValueError('Invalid ellipsoid factors: unrepresentable axis ratio')
    transformed_delta = delta[positions] @ transform.T
    coefficients = np.einsum('bji,bj->bi',eigenvectors,transformed_delta)**2
    if diagnostics is not None:
        diagnostics['svd_contact_pairs'] += len(positions)
    low, high = np.zeros(len(positions)), np.ones(len(positions))
    tolerance = 1+1e-9
    for _ in range(48):
        t = (low+high)*.5
        denominator = 1-t[:,None]+t[:,None]*eigenvalues
        derivative = np.sum(coefficients*((1-t[:,None])**2-eigenvalues*t[:,None]**2)/denominator**2,axis=1)
        contact = t*(1-t)*np.sum(coefficients/denominator,axis=1)
        # Concavity: the tangent gives an upper bound on the maximum over
        # the bracket. Any value > tolerance proves separation immediately.
        upper = contact + derivative*np.where(derivative>0,high-t,low-t)
        margin = 1e-12*np.maximum(1,np.abs(contact))
        yes = upper+margin <= tolerance
        no = contact-margin > tolerance
        accepted[positions[yes]] = True
        active = ~(yes|no)
        if not active.any():
            return accepted
        low = np.where(derivative>0,t,low)[active]
        high = np.where(derivative>0,high,t)[active]
        positions = positions[active]
        eigenvalues,coefficients = eigenvalues[active],coefficients[active]
    t = (low+high)*.5
    contact = t*(1-t)*np.sum(coefficients/(1-t[:,None]+t[:,None]*eigenvalues),axis=1)
    accepted[positions] = contact <= tolerance
    return accepted
