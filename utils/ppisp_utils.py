"""PPISP bridge: stable view identities, optimization state, and CHW rendering."""
from dataclasses import asdict
from pathlib import Path

import torch


def view_key(camera):
    return str(getattr(camera, "ppisp_frame_name", getattr(camera, "image_name", "")))


class PPISPAdapter:
    def __init__(self, cameras, iterations, payload=None):
        from ppisp import PPISP, PPISPConfig
        self.iterations = iterations
        self.controller_steps = 0 if payload is None else payload["controller_steps"]
        camera_ids = sorted({int(c.ppisp_camera_id) for c in cameras})
        self.camera_ids = {c: i for i, c in enumerate(camera_ids)}
        names = sorted(view_key(c) for c in cameras)
        if len(names) != len(set(names)):
            raise ValueError("PPISP requires unique training image names.")
        self.frames = {name: i for i, name in enumerate(names)}
        if not names:
            raise ValueError("PPISP requires training cameras.")
        config = PPISPConfig(controller_activation_ratio=10_000 / 16_000)
        if payload is not None:
            if payload["camera_ids"] != self.camera_ids or payload["frames"] != self.frames:
                raise ValueError("PPISP checkpoint camera/frame mapping differs from dataset.")
            if payload["iterations"] != iterations:
                raise ValueError("Resume with the original --iterations for PPISP scheduling.")
            config = PPISPConfig(**payload["config"])
        self.module = PPISP(len(self.camera_ids), len(self.frames), config)
        self.optimizers = self.module.create_optimizers()
        self.schedulers = self.module.create_schedulers(self.optimizers, iterations)
        if payload is not None:
            self.module.load_state_dict(payload["model"])
            for opt, state in zip(self.optimizers, payload["optimizers"]):
                opt.load_state_dict(state)
            for sched, state in zip(self.schedulers, payload["schedulers"]):
                sched.load_state_dict(state)

    @property
    def distilling(self):
        cfg = self.module.config
        return (cfg.use_controller and cfg.controller_distillation
                and self.schedulers[0].last_epoch >= int(cfg.controller_activation_ratio * self.iterations))

    def apply(self, image, camera):
        # Viewer cameras have no dataset identity; use the first learned camera.
        physical_id = getattr(camera, "ppisp_camera_id", None)
        if physical_id is None:
            camera_idx = 0
        elif int(physical_id) not in self.camera_ids:
            raise ValueError(f"PPISP has no trained physical camera {physical_id}.")
        else:
            camera_idx = self.camera_ids[int(physical_id)]
        frame_idx = self.frames.get(view_key(camera), -1)
        rgb = image.permute(1, 2, 0).contiguous()
        return self.module(rgb, camera_idx=camera_idx, frame_idx=frame_idx).permute(2, 0, 1).contiguous()

    def step(self):
        if self.distilling:
            self.controller_steps += 1
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()

    def capture(self):
        return dict(version=1, iterations=self.iterations, controller_steps=self.controller_steps,
                    camera_ids=self.camera_ids, frames=self.frames,
                    config=asdict(self.module.config), model=self.module.state_dict(),
                    optimizers=[o.state_dict() for o in self.optimizers],
                    schedulers=[s.state_dict() for s in self.schedulers])

    def save(self, model_path, iteration):
        path = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "ppisp.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.capture(), path)

    @classmethod
    def load(cls, model_path, iteration, cameras):
        path = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "ppisp.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing PPISP state: {path}")
        payload = torch.load(path, map_location="cuda", weights_only=False)
        adapter = cls(cameras, payload["iterations"], payload)
        # A snapshot exactly at activation has not trained the controller yet.
        if adapter.controller_steps == 0:
            adapter.schedulers[0].last_epoch = min(
                adapter.schedulers[0].last_epoch,
                int(adapter.module.config.controller_activation_ratio * adapter.iterations) - 1)
        adapter.module.eval()
        return adapter
