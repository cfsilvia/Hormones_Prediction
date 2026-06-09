import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def simplex_ls(Z, x):
    """min_a ||x - Z.T @ a||^2  s.t.  a >= 0, sum(a) = 1"""
    k = Z.shape[0]
    cons = {'type': 'eq', 'fun': lambda a: a.sum() - 1.0}
    bounds = [(0, 1)] * k
    a0 = np.ones(k) / k
    res = minimize(lambda a: np.sum((x - Z.T @ a) ** 2),
                   a0, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 300, 'ftol': 1e-12})
    return res.x


def project_onto_convex_polygon(p, hull):
    """Project 2D point p onto convex polygon defined by hull."""
    verts = hull.points[hull.vertices]
    nv = len(verts)
    for i in range(nv):
        v0, v1 = verts[i], verts[(i + 1) % nv]
        if np.cross(v1 - v0, p - v0) > 1e-12:
            break
    else:
        return p

    best_dist = np.inf
    best_proj = p.copy()
    for i in range(nv):
        v0, v1 = verts[i], verts[(i + 1) % nv]
        edge = v1 - v0
        t = np.clip(np.dot(p - v0, edge) / np.dot(edge, edge), 0, 1)
        proj = v0 + t * edge
        dist = np.sum((p - proj) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_proj = proj
    return best_proj


def archetypal_analysis(X, k, hull=None, max_iter=100, tol=1e-6, verbose=True):
    """Archetypal analysis in 2D with hull-projected archetypes.

    Returns Z (k x 2 archetype coordinates), A (n x k mixing weights), error.
    """
    n, m = X.shape
    rng = np.random.default_rng(42)

    if hull is None:
        hull = ConvexHull(X)

    vertices = hull.vertices
    if k <= len(vertices):
        idx = rng.choice(vertices, k, replace=False)
    else:
        idx = rng.choice(n, k, replace=False)
    Z = X[idx].copy().astype(float)
    A = rng.dirichlet(np.ones(k), size=n)

    prev_err = np.inf
    for it in range(max_iter):
        for i in range(n):
            A[i] = simplex_ls(Z, X[i])
        Z_new = np.linalg.lstsq(A, X, rcond=None)[0]
        for j in range(k):
            Z_new[j] = project_onto_convex_polygon(Z_new[j], hull)
        Z = Z_new
        err = np.sum((X - A @ Z) ** 2)
        delta = abs(prev_err - err)
        if verbose:
            print(f"  Iter {it+1:3d}  error = {err:.4f}  delta = {delta:.2e}")
        if delta < tol:
            break
        prev_err = err

    return Z, A, err
