"""Joint environment/SfM initialization and GCISP controller training."""
import math
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud, fov2focal


def env_config(opt):
    return {k:v for k,v in vars(opt).items() if k.startswith('env_')}


def validate_environment_options(dataset, opt):
    if not dataset.hdr_camera:
        raise ValueError('--env_background requires --hdr_camera')
    if not math.isfinite(opt.densify_grad_threshold) or opt.densify_grad_threshold <= 0:
        raise ValueError('Require positive densify_grad_threshold so environment rows remain excluded')
    if opt.optimizer_type != 'default':
        raise ValueError('Environment training requires --optimizer_type default')
    durations = [opt.env_scaled_iterations, opt.env_joint_iterations, opt.env_controller_iterations]
    if any(n <= 0 for n in durations) or sum(durations) != opt.iterations:
        raise ValueError('--iterations must equal the three positive environment stage lengths')
    if not 0 < opt.env_train_iterations <= opt.env_scaled_iterations + opt.env_joint_iterations:
        raise ValueError('env_train_iterations must end before controller-only training')
    if opt.env_points < 4 or not math.isfinite(opt.env_radius_pixel) or opt.env_radius_pixel <= 0:
        raise ValueError('Require env_points >= 4 and finite env_radius_pixel > 0')
    if not math.isfinite(opt.env_min_gradient) or opt.env_min_gradient < 0 or opt.env_min_gradient_hits < 1:
        raise ValueError('Require nonnegative env_min_gradient and positive env_min_gradient_hits')
    if opt.disable_gradient_scaling or opt.gradient_scale_anneal_end != opt.env_scaled_iterations:
        raise ValueError('Enable gradient scaling and match its end to env_scaled_iterations')


def phase_at(iteration, opt):
    if iteration <= opt.env_scaled_iterations:
        return 'sfm_scaled', iteration
    if iteration <= opt.env_scaled_iterations + opt.env_joint_iterations:
        return 'joint', iteration
    return 'controller', iteration


def parallax_sphere_radius(centers, focal_lengths, parallax_pixels):
    """Nominal R=f_max*B_max/p; not a strict all-view pixel-disparity bound."""
    centers = np.asarray(centers, dtype=np.float64)
    focals = np.asarray(focal_lengths, dtype=np.float64)
    if (centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 2
            or not np.isfinite(centers).all() or not focals.size
            or not np.isfinite(focals).all() or (focals <= 0).any()
            or not math.isfinite(parallax_pixels) or parallax_pixels <= 0):
        raise ValueError('Parallax radius requires finite camera centers, positive focals and p')
    # Chunk the exact pairwise maximum to avoid an NxNx3 allocation.
    baseline = max(float(np.linalg.norm(centers[i:i+128,None,:]-centers[None,:,:],axis=-1).max())
                   for i in range(0,len(centers),128))
    max_focal = float(focals.max())
    radius = max_focal * baseline / parallax_pixels
    center = centers.mean(axis=0)
    if not math.isfinite(radius) or radius <= np.linalg.norm(centers-center,axis=1).max():
        raise ValueError('Parallax sphere cannot enclose cameras: zero baseline or p too large')
    return radius, baseline, max_focal


def fibonacci_sphere(count, center, radius):
    i = np.arange(count, dtype=np.float64)
    z = 1 - 2*(i+.5)/count
    angle = i*(math.pi*(3-math.sqrt(5)))
    xy = np.sqrt(1-z*z)
    return (np.asarray(center)[None]+radius*np.stack((xy*np.cos(angle),xy*np.sin(angle),z),axis=1)).astype(np.float32)


class EnvironmentGaussianModel(GaussianModel):
    """Rotation is fixed for the first env_train_iterations; the full prefix freezes afterwards."""
    def __init__(self, sh_degree, optimizer_type, opt):
        super().__init__(sh_degree,optimizer_type)
        self.env_opt = opt
        self.env_count = 0
        self.env_sh_degree = 0
        self.sfm_added = False
        self.env_pruned = False
        self.allow_env_pruning = False
        self.env_gradient_sum = None
        self.env_gradient_hits = None
        self.env_gradient_steps = 0

    def initialize_environment(self, pcd, cam_infos, extent, cameras):
        self.sfm_pcd = pcd
        self.sfm_cam_infos = cam_infos
        centers = torch.stack([c.camera_center for c in cameras]).detach().cpu().numpy()
        center = centers.mean(axis=0)
        focals = [max(fov2focal(c.FoVx,c.image_width),fov2focal(c.FoVy,c.image_height))
                  for c in cameras]
        radius, baseline, max_focal = parallax_sphere_radius(
            centers, focals, self.env_opt.env_radius_pixel)
        points = fibonacci_sphere(self.env_opt.env_points,center,radius)
        shell = BasicPointCloud(points=points,colors=np.full_like(points,.5),normals=np.zeros_like(points))
        self.create_from_pcd(shell,cam_infos,extent)
        with torch.no_grad():
            self._scaling.fill_(math.log(radius*max(math.sqrt(4*math.pi/len(points)), 1e-6)))
            self._opacity.fill_(math.log(.2/.8))
        self.env_count = len(points)
        self.env_gradient_sum = torch.zeros(self.env_count,device='cuda')
        self.env_gradient_hits = torch.zeros(self.env_count,dtype=torch.long,device='cuda')
        print('Environment shell: {} points, baseline {}, focal {} px, p {} px, radius {}'.format(
            self.env_count, baseline, max_focal, self.env_opt.env_radius_pixel, radius))

    def set_camera_stage(self, iteration):
        active = phase_at(iteration,self.env_opt)[0] == 'controller'
        self.camera_model.set_stage(1 if active else 0,1)
        self.controller_stage = active
        for tensor in (self._xyz,self._features_dc,self._features_rest,self._scaling,self._rotation,self._opacity):
            tensor.requires_grad_(not active)
        return active

    def environment_learning_rates(self, iteration):
        return dict(xyz=1e-4 * (.01 ** min(iteration / 30000., 1.)),
                    f_dc=.0025, f_rest=.0025/20., opacity=.01,
                    scaling=.0001, rotation=0.)

    @torch.no_grad()
    def step_joint_optimizer(self, iteration):
        # Adam moments are elementwise and independent of LR. Correct only the
        # environment delta so shared tensors can use distinct scene/env LRs.
        saved = []
        if iteration <= self.env_opt.env_train_iterations:
            rates = self.environment_learning_rates(iteration)
            for group in self.optimizer.param_groups:
                parameter = group['params'][0]
                target_lr = rates[group['name']]
                if group['lr'] == 0 and target_lr != 0:
                    raise ValueError('A zero scene LR cannot be combined with a nonzero environment LR')
                if group['lr'] != 0 and target_lr != group['lr']:
                    saved.append((parameter, parameter[:self.env_count].clone(), target_lr/group['lr']))
        self.optimizer.step()
        for parameter, before, ratio in saved:
            parameter[:self.env_count].copy_(before + ratio*(parameter[:self.env_count]-before))

    @torch.no_grad()
    def freeze_environment_parameters(self, iteration):
        warmup = iteration <= self.env_opt.env_train_iterations
        parameters = ((self._rotation,) if warmup else
            (self._xyz, self._scaling, self._rotation, self._opacity,
             self._features_dc, self._features_rest))
        # Clear both gradients and Adam moments, so resumed momentum cannot
        # update frozen environment rows. SfM rows remain trainable.
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad[:self.env_count] = 0
            state = self.optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    state[key][:self.env_count] = 0

    @torch.no_grad()
    def record_environment_gradients(self):
        if self._features_dc.grad is None:
            return
        magnitude = self._features_dc.grad[:self.env_count].flatten(1).norm(dim=1)
        magnitude = torch.where(torch.isfinite(magnitude),magnitude,torch.zeros_like(magnitude))
        self.env_gradient_sum += magnitude
        self.env_gradient_hits += (magnitude>self.env_opt.env_min_gradient).long()
        self.env_gradient_steps += 1

    @torch.no_grad()
    def prune_untrained_environment(self):
        if self.env_pruned:
            return
        mean = self.env_gradient_sum/max(1,self.env_gradient_steps)
        remove_env = (mean<=self.env_opt.env_min_gradient) | (self.env_gradient_hits<self.env_opt.env_min_gradient_hits)
        remove = torch.zeros(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
        remove[:self.env_count] = remove_env
        before = self.env_count
        self.tmp_radii = torch.zeros(len(self._xyz),device=self._xyz.device)
        self.allow_env_pruning = True
        try:
            self.prune_points(remove)
        finally:
            self.allow_env_pruning = False
            self.tmp_radii = None
        self.env_pruned = True
        print('Environment training complete: retained {} / {}'.format(self.env_count,before))

    @torch.no_grad()
    def add_sfm_points(self):
        if self.sfm_added:
            return
        fresh = GaussianModel(self.max_sh_degree,self.optimizer_type)
        fresh.create_from_pcd(self.sfm_pcd,self.sfm_cam_infos,self.spatial_lr_scale)
        self.tmp_radii = torch.zeros(len(self._xyz),device=self._xyz.device)
        self.densification_postfix(fresh._xyz.detach(),fresh._features_dc.detach(),fresh._features_rest.detach(),
            fresh._opacity.detach(),fresh._scaling.detach(),fresh._rotation.detach(),
            torch.zeros(len(fresh._xyz),device=self._xyz.device))
        self.tmp_radii = None
        self.sfm_added = True
        print('SfM initialized: {} scene + {} environment points'.format(len(fresh._xyz),self.env_count))

    def evaluate_sh(self, shs, directions):
        from utils.sh_utils import eval_sh
        return eval_sh(self.active_sh_degree, shs, directions)

    def prune_points(self, mask):
        mask = mask.clone()
        if not self.allow_env_pruning:
            mask[:self.env_count] = False
        keep = ~mask[:self.env_count]
        super().prune_points(mask)
        if self.env_gradient_sum is not None:
            self.env_gradient_sum = self.env_gradient_sum[keep]
            self.env_gradient_hits = self.env_gradient_hits[keep]
        self.env_count = int(keep.sum().item())

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grads = grads.clone(); grads[:self.env_count] = 0
        super().densify_and_clone(grads,grad_threshold,scene_extent)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        grads = grads.clone(); grads[:self.env_count] = 0
        super().densify_and_split(grads,grad_threshold,scene_extent,N)

    def reset_opacity(self):
        old = self._opacity[:self.env_count].detach().clone()
        super().reset_opacity()
        with torch.no_grad():
            self._opacity[:self.env_count].copy_(old)

    def capture(self):
        payload = super().capture()
        payload['environment'] = dict(version=2,config=env_config(self.env_opt),count=self.env_count,
            sfm_added=self.sfm_added,pruned=self.env_pruned,sh_degree=self.env_sh_degree,
            gradient_sum=self.env_gradient_sum,gradient_hits=self.env_gradient_hits,
            gradient_steps=self.env_gradient_steps)
        return payload

    def restore(self, model_args, training_args):
        state = model_args.get('environment') if isinstance(model_args,dict) else None
        if state is None or state.get('version')!=2 or state.get('config')!=env_config(self.env_opt):
            raise ValueError('Resume requires a joint environment v2 checkpoint with identical options; pretraining checkpoints are incompatible')
        super().restore(model_args,training_args)
        self.env_count = state['count']
        self.sfm_added = state['sfm_added']
        self.env_pruned = state['pruned']
        self.env_sh_degree = state['sh_degree']
        self.env_gradient_sum = state['gradient_sum'].to(self._xyz.device)
        self.env_gradient_hits = state['gradient_hits'].to(self._xyz.device)
        self.env_gradient_steps = state['gradient_steps']
