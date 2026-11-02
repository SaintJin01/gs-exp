"""CUDA integration checks; run from the repository with the PPISP env active."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.ppisp_utils import PPISPAdapter


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    cameras = [SimpleNamespace(image_name=n, ppisp_camera_id=c)
               for n, c in [("b.png", 7), ("a.png", 7), ("c.png", 19)]]
    isp = PPISPAdapter(cameras, 10)
    assert isp.frames == {"a.png": 0, "b.png": 1, "c.png": 2}
    assert isp.camera_ids == {7: 0, 19: 1}
    for i in range(10):
        raw = torch.rand(3, 18, 24, device="cuda", requires_grad=True)
        prediction = isp.apply(raw, cameras[i % 3])
        assert prediction.shape == raw.shape
        loss = prediction.square().mean()
        if not isp.distilling:
            loss = loss + isp.module.get_regularization_loss()
        loss.backward()
        if i < 8:
            assert raw.grad is not None and torch.isfinite(raw.grad).all()
        else:
            assert raw.grad is None, "Scene must be detached during distillation"
            assert any(p.grad is not None for p in isp.module.controllers.parameters())
        isp.step()
        if i == 7:
            with tempfile.TemporaryDirectory() as tmp:
                isp.save(tmp, 8)
                restored = PPISPAdapter.load(tmp, 8, cameras[::-1])
                assert not restored.distilling, "Untrained controller must not run at activation boundary"
    assert isp.controller_steps == 2
    raw = torch.rand(3, 18, 24, device="cuda")
    novel = SimpleNamespace(image_name="novel.png", ppisp_camera_id=7)
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        expected = isp.apply(raw, novel)
        isp.save(tmp, 10)
        restored = PPISPAdapter.load(tmp, 10, cameras[::-1])
        torch.testing.assert_close(expected, restored.apply(raw, novel))
        resumed = PPISPAdapter(cameras[::-1], 10, isp.capture())
        assert resumed.distilling and resumed.controller_steps == 2
        assert resumed.schedulers[0].state_dict() == isp.schedulers[0].state_dict()
        torch.testing.assert_close(expected, resumed.apply(raw, novel))
        try:
            PPISPAdapter(cameras, 11, isp.capture())
        except ValueError:
            pass
        else:
            raise AssertionError("Changed training schedule must be rejected")
    try:
        isp.apply(raw, SimpleNamespace(image_name="x", ppisp_camera_id=99))
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown physical camera must be rejected")
    print("PASS: CUDA backward, controller freeze, stable mapping, snapshot and resume")


if __name__ == "__main__":
    main()
