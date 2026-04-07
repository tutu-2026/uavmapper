import pyceres
import pycolmap

def define_problem(rec):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        for p in im.points2D:
            cost = pycolmap.cost_functions.ReprojErrorCost(
                cam.model, im.cam_from_world, p.xy
            )
            prob.add_residual_block(
                cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params]
            )
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    return prob


def solve(prob):
    print(
        prob.num_parameter_blocks(),
        prob.num_parameters(),
        prob.num_residual_blocks(),
        prob.num_residuals(),
    )
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())

    