"""Runtime API (``motionbricks_mlx.runtime``) against upstream's own runtime code.

Three layers:
* weight-free unit tests of the transforms, tokens, recorder, buffers, settings and player timing;
* teacher-forced parity with ``sandbox/golden/runtime`` (recorded from upstream's Warp kernels by
  ``sandbox/scripts/make_runtime_golden.py``): clip loading, encoding, decoding, endpoint
  correction, recorder sampling, target placement, speed scaling, IK pins and sampling;
* end-to-end generation with the MLX model on the same scenarios (needs the checkpoints).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from motionbricks_mlx import runtime as rt
from motionbricks_mlx.runtime import transforms as T
from mlx_helpers import GOLDEN, ROOT, load_model

RUNTIME_GOLDEN = Path(os.environ.get("MOTIONBRICKS_RUNTIME_GOLDEN", GOLDEN / "runtime"))
# Clips: `motionbricks-mlx download --only-clips` (checkpoints/motions), the sandbox's upstream
# checkout, or $MOTIONBRICKS_CLIPS.
_SANDBOX_CLIPS = ROOT / "sandbox" / "upstream" / "assets" / "motions"
CLIPS = Path(os.environ.get("MOTIONBRICKS_CLIPS", ROOT / "checkpoints" / "motions"
                            if (ROOT / "checkpoints" / "motions").is_dir() else _SANDBOX_CLIPS))
SCENARIOS = ("walk", "jog_one_pose", "no_correction", "model_to_runtime")
RNG = np.random.default_rng(1)


def random_quat(n):
    q = RNG.normal(size=(n, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def same_transforms(a, b, atol):
    """Max error between transform arrays, treating q and -q as the same rotation."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    qa, qb = a[..., 3:7], b[..., 3:7]
    sign = np.where((qa * qb).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    err = max(np.abs(a[..., :3] - b[..., :3]).max(initial=0), np.abs(qa * sign - qb).max(initial=0))
    assert err < atol, f"max error {err}"
    return err


def same_dof(a, b, atol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    same_transforms(a[..., :7], b[..., :7], atol)
    err = np.abs(a[..., 7:] - b[..., 7:]).max(initial=0)
    assert err < atol, f"joint angles max error {err}"


# --- weight-free -------------------------------------------------------------------------------


def test_transform_algebra():
    q = random_quat(20)
    t = T.transform(RNG.normal(size=(20, 3)), q)
    ident = T.transform_mul(t, T.transform_inverse(t))
    assert np.abs(ident[:, :3]).max() < 1e-5 and np.abs(np.abs(ident[:, 6]) - 1).max() < 1e-5
    p = RNG.normal(size=(20, 3)).astype(np.float32)
    np.testing.assert_allclose(T.quat_rotate_inv(q, T.quat_rotate(q, p)), p, atol=1e-5)
    m = np.stack([T.quat_rotate(q, e) for e in np.eye(3, dtype=np.float32)], -1)
    back = T.quat_from_matrix(m)
    assert np.abs(np.abs((back * q).sum(-1)) - 1).max() < 1e-5
    half = T.quat_slerp(T.IDENTITY_QUAT, q, 0.5)
    np.testing.assert_allclose(np.abs((T.quat_mul(half, half) * q).sum(-1)), 1, atol=1e-5)
    np.testing.assert_allclose(T.quadratic_fade_weight([0.0, 0.25, 0.5, 1.0]), [1.0, 0.875, 0.5, 0.0], atol=1e-6)


def test_valid_frame_count_and_config():
    assert [rt.valid_frame_count(n) for n in (6, 10, 16)] == [18, 34, 58]
    with pytest.raises(ValueError):
        rt.valid_frame_count(5)
    assert rt.MotionBricksInferenceConfig().allowed_token_bits() == 0b00001111111 << 2
    for bad in ({"allowed_token_counts": ()}, {"allowed_token_counts": (5,)}, {"heading_alignment": (0.0, 0.0)}):
        with pytest.raises(ValueError):
            rt.MotionBricksInferenceConfig(**bad)


def test_tokens_pad_with_the_last_pose():
    token = rt.DOFToken(29)
    poses = RNG.normal(size=(2, 36)).astype(np.float32)
    token.update_from_poses(poses)
    assert token.num_poses == 2 and np.array_equal(token.poses[2], poses[1]) and np.array_equal(token.poses[3], poses[1])
    token.update_from_poses(poses[0])
    assert token.num_poses == 1 and np.array_equal(token.poses, np.broadcast_to(poses[0], (4, 36)))
    with pytest.raises(ValueError):
        token.update_from_poses(np.zeros((5, 36), np.float32))


def test_recorder_and_resampling():
    history = rt.MotionRecorder(np.zeros(36, np.float32), capacity=4)
    for k in range(1, 6):
        history.record(1 / 30, np.full(36, k, np.float32))
    assert history.poses[:, 10].tolist() == [2, 3, 4, 5] and history.delta_times[-1] == 0
    lower, upper, blend = rt.history_sample_indices(np.array([1 / 60] * 7 + [0], np.float32), 4, 30.0)
    # Poses every 1/60 s; samples every 1/30 s ending at the latest pose (blend 1 toward it).
    assert lower.tolist() == [1, 3, 5, 6] and upper.tolist() == [2, 4, 6, 7]
    np.testing.assert_allclose(blend, [0, 0, 0, 1], atol=1e-5)
    with pytest.raises(ValueError):
        history.record(0.0, np.zeros(36, np.float32))


def test_buffer_sampling_clamps_and_blends():
    poses = np.zeros((3, 36), np.float32)
    poses[:, 6] = 1.0
    poses[:, 10] = [0.0, 1.0, 2.0]
    buffer = rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, 30.0, poses, foot_contacts=np.eye(3, 4, dtype=np.float32))
    assert buffer.sample(0.5 / 30)[10] == pytest.approx(0.5) and buffer.sample(99.0)[10] == 2.0 and buffer.sample(-1.0)[10] == 0.0
    np.testing.assert_allclose(buffer.sample_foot_contacts(1.5 / 30), [0, 0.5, 0.5, 0], atol=1e-6)
    assert buffer.duration == pytest.approx(2 / 30)


class _FakeEngine:
    """Generates a fixed-length motion, so the player's timing can be checked without a model."""

    def __init__(self, frames):
        self.frames, self.calls = frames, 0

    def create_input(self):
        return rt.DOFToken(29)

    def create_output(self):
        return rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, 30.0, np.tile(rt.transforms.IDENTITY.tolist() + [0.0] * 29, (58, 1)),
                                     num_frames=0, foot_contacts=np.zeros((58, 4), np.float32))

    def generate_animation(self, context, target, output, config=None):
        self.calls += 1
        self.config = config
        output.num_frames = self.frames


def test_player_timing_and_overrides():
    engine = _FakeEngine(31)
    reference = np.zeros(36, np.float32)
    reference[6] = 1.0
    player = rt.GeneratedDOFMotionPlayer(engine, reference)
    player.seed_history_from_pose(reference)
    params = rt.TargetPoseInferenceParams(transition_early_by_frames=10, num_poses_to_use=2, allowed_token_counts=(9,))
    player.generate(np.tile(reference, (4, 1)), params)
    assert player.target.num_poses == 2 and engine.config.allowed_token_counts == (9,)
    assert player.time_to_transition == pytest.approx(20 / 30)
    pose, contacts = np.zeros(36, np.float32), np.zeros(4, np.float32)
    player.sample(0.5, pose, contacts)
    assert player.time_to_transition == pytest.approx(20 / 30 - 0.5)
    player.sample(1.0, pose, contacts)
    assert player.time_to_transition == 0.0


class _FlakyEngine(_FakeEngine):
    """Writes the call number into each frame's root X, and raises while ``fail`` is set."""

    fail = False

    def generate_animation(self, context, target, output, config=None):
        if self.fail:
            output.num_frames = 0  # like the real engines, which clear the output first
            raise RuntimeError("generation failed")
        super().generate_animation(context, target, output, config)
        output.poses[: self.frames, 0] = self.calls


def test_player_keeps_its_motion_when_generation_fails():
    engine = _FlakyEngine(31)
    reference = np.zeros(36, np.float32)
    reference[6] = 1.0
    player = rt.GeneratedDOFMotionPlayer(engine, reference)
    player.seed_history_from_pose(reference)
    player.generate(np.tile(reference, (4, 1)), rt.TargetPoseInferenceParams(transition_early_by_frames=5, num_poses_to_use=4))
    pose, contacts = np.zeros(36, np.float32), np.zeros(4, np.float32)
    player.sample(0.25, pose, contacts)
    before = (player.output, player.playback_time, player.time_to_transition, player.current_root_position().copy())
    engine.fail = True
    with pytest.raises(RuntimeError):
        player.generate(np.tile(reference, (4, 1)), rt.TargetPoseInferenceParams(transition_early_by_frames=12, num_poses_to_use=4))
    assert player.output is before[0] and player.output.num_frames == 31 and player.transition_early_by_frames == 5
    assert player.playback_time == before[1] and player.time_to_transition == before[2]
    assert np.array_equal(player.current_root_position(), before[3])
    player.sample(0.1, pose, contacts)  # still plays the first motion
    assert pose[0] == 1.0
    engine.fail = False
    player.generate(np.tile(reference, (4, 1)), rt.TargetPoseInferenceParams(transition_early_by_frames=12, num_poses_to_use=4))
    assert player.output is not before[0] and player.transition_early_by_frames == 12 and player.playback_time == 0.0
    player.sample(0.0, pose, contacts)
    assert pose[0] == 2.0  # the motion of the second successful generation


def _quat_from_matrix_per_matrix(m):
    """The previous per-matrix port of ``wp.quat_from_matrix``: the reference for the vectorized one."""
    f = np.float32
    shape, m = m.shape[:-2], np.asarray(m, f).reshape(-1, 3, 3)
    out = np.empty((m.shape[0], 4), f)
    for i, r in enumerate(m):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr >= 0:
            h = np.sqrt(tr + f(1))
            w, h = f(0.5) * h, f(0.5) / h
            x, y, z = (r[2, 1] - r[1, 2]) * h, (r[0, 2] - r[2, 0]) * h, (r[1, 0] - r[0, 1]) * h
        else:
            k = 0
            if r[1, 1] > r[0, 0]:
                k = 1
            if r[2, 2] > r[k, k]:
                k = 2
            if k == 0:
                h = np.sqrt((r[0, 0] - (r[1, 1] + r[2, 2])) + f(1))
                x, h = f(0.5) * h, f(0.5) / h
                y, z, w = (r[0, 1] + r[1, 0]) * h, (r[2, 0] + r[0, 2]) * h, (r[2, 1] - r[1, 2]) * h
            elif k == 1:
                h = np.sqrt((r[1, 1] - (r[2, 2] + r[0, 0])) + f(1))
                y, h = f(0.5) * h, f(0.5) / h
                z, x, w = (r[1, 2] + r[2, 1]) * h, (r[0, 1] + r[1, 0]) * h, (r[0, 2] - r[2, 0]) * h
            else:
                h = np.sqrt((r[2, 2] - (r[0, 0] + r[1, 1])) + f(1))
                z, h = f(0.5) * h, f(0.5) / h
                x, y, w = (r[2, 0] + r[0, 2]) * h, (r[1, 2] + r[2, 1]) * h, (r[1, 0] - r[0, 1]) * h
        out[i] = (x, y, z, w)
    return T.quat_normalize(out).reshape((*shape, 4))


def test_quat_from_matrix_takes_warps_branches_bitwise():
    q = random_quat(2000)
    sweeps = [T.quat_from_axis_angle(axis, np.linspace(-np.pi, np.pi, 301, dtype=np.float32)) for axis in np.eye(3, dtype=np.float32)]
    for quats in (q, *sweeps):  # every branch: trace >= 0, and the X, Y and Z diagonal cases
        m = np.stack([T.quat_rotate(quats, e) for e in np.eye(3, dtype=np.float32)], -1)
        assert np.array_equal(T.quat_from_matrix(m), _quat_from_matrix_per_matrix(m))
    noisy = RNG.normal(size=(7, 5, 3, 3)).astype(np.float32)  # not rotations; zero matrices too
    noisy[0] = 0.0
    assert np.array_equal(T.quat_from_matrix(noisy), _quat_from_matrix_per_matrix(noisy))


def test_cross_matches_numpy_bitwise():
    for shape_a, shape_b in (((3,), (3,)), ((40, 3), (40, 3)), ((4, 9, 3), (3,)), ((1, 3), (6, 3))):
        a, b = RNG.normal(size=shape_a).astype(np.float32), RNG.normal(size=shape_b).astype(np.float32)
        assert np.array_equal(T.cross(a, b), np.cross(a, b))


def random_hierarchy(num_joints):
    return np.array([-1] + [int(RNG.integers(0, j)) for j in range(1, num_joints)], np.int32)


def test_world_poses_by_depth_match_joint_by_joint():
    parents = random_hierarchy(50)
    local = T.transform(RNG.normal(size=(3, 50, 3)), random_quat(150).reshape(3, 50, 4))
    root = T.transform([0.5, -2.0, 1.0], random_quat(1)[0])
    for poses in (local, local[0]):
        want = np.empty_like(poses)
        want[..., 0, :] = T.transform_mul(root, poses[..., 0, :])
        for j in range(1, 50):
            want[..., j, :] = T.transform_mul(want[..., parents[j], :], poses[..., j, :])
        assert np.array_equal(T.compute_world_poses(poses, parents, root), want)
        skeleton = rt.Skeleton([f"j{j}" for j in range(50)], parents, local[0])
        assert np.array_equal(skeleton.compute_world_transforms(poses, root), want)


def test_skeleton_rejects_hierarchies_out_of_order():
    reference = np.tile(T.IDENTITY, (3, 1))
    for parents, joint in (([-1, 2, 0], "Joint 1"), ([-1, -1, 0], "Joint 1"), ([-1, 0, 2], "Joint 2"), ([0, 0, 1], "Joint 0")):
        with pytest.raises(ValueError, match=joint):
            rt.Skeleton(["a", "b", "c"], parents, reference)
    skeleton = rt.Skeleton(["a", "b", "c"], [-1, 0, 1], reference)
    with pytest.raises(ValueError):
        skeleton.parent_indices[1] = 2  # engines cache mappings built from it


def _loop_frame_by_frame(buffer):
    """The previous per-frame ``make_loopable``: the reference for the vectorized one."""
    p, last = buffer.poses, buffer.num_frames - 1
    if isinstance(buffer, rt.AnimationBuffer):
        root_delta = T.compute_root_endpoint_delta(p[last, 0], p[0, 0])
        deltas = T.compute_delta_transform(p[last, 1:], p[0, 1:])
        for frame in range(buffer.num_frames):
            w = np.float32(frame / last)
            p[frame, 0] = T.apply_root_endpoint_correction(p[frame, 0], T.IDENTITY, root_delta, w)
            p[frame, 1:] = T.apply_delta_transform(p[frame, 1:], deltas, np.full(len(deltas), w, np.float32))
        return
    from motionbricks_mlx.runtime.animation import read_root, write_root

    root_delta = T.compute_root_endpoint_delta(read_root(p[last]), read_root(p[0]))
    angle_deltas = p[0, 7:] - p[last, 7:]
    for frame in range(buffer.num_frames):
        w = np.float32(frame / last)
        write_root(T.apply_root_endpoint_correction(read_root(p[frame]), T.IDENTITY, root_delta, w), p[frame])
        p[frame, 7:] += angle_deltas * w


def test_make_loopable_matches_frame_by_frame():
    skeleton = rt.Skeleton([f"j{j}" for j in range(6)], random_hierarchy(6), np.tile(T.IDENTITY, (6, 1)))
    soma = T.transform(RNG.normal(size=(45, 6, 3)), random_quat(270).reshape(45, 6, 4))
    dof = np.concatenate([RNG.normal(size=(45, 3)), random_quat(45), RNG.normal(size=(45, 29))], -1)
    for make in (lambda: rt.AnimationBuffer(skeleton, 30.0, soma.copy(), num_frames=40),
                 lambda: rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, 30.0, dof.copy(), num_frames=40)):
        got, want = make(), make()
        got.make_loopable()
        _loop_frame_by_frame(want)
        assert np.array_equal(got.poses, want.poses)  # frames past num_frames stay untouched too


def test_tokens_from_recorder_match_per_sample_blending():
    from motionbricks_mlx.runtime.animation import read_root, write_root

    dts = np.array([1 / 45, 1 / 50, 1 / 61, 1 / 30, 1 / 75, 1 / 40, 1 / 60], np.float32)
    soma = rt.MotionRecorder(T.transform(RNG.normal(size=(5, 3)), random_quat(5)), capacity=8)
    dof = rt.MotionRecorder(np.concatenate([RNG.normal(size=3), random_quat(1)[0], RNG.normal(size=29)]), capacity=8)
    for dt in dts:
        soma.record(float(dt), T.transform(RNG.normal(size=(5, 3)), random_quat(5)))
        dof.record(float(dt), np.concatenate([RNG.normal(size=3), random_quat(1)[0], RNG.normal(size=29)]))
    lower, upper, blends = rt.history_sample_indices(soma.delta_times, 4, 30.0)
    pose_token, dof_token = rt.PoseToken(5), rt.DOFToken(29)
    pose_token.update_from_recorder(soma)
    dof_token.update_from_recorder(dof)
    for k in range(4):
        h = soma.poses
        assert np.array_equal(pose_token.poses[k], T.blend_transform(h[lower[k]], h[upper[k]], np.full(5, blends[k], np.float32)))
        a, b, w = dof.poses[lower[k]], dof.poses[upper[k]], np.float32(blends[k])
        want = (a + (b - a) * w).astype(np.float32)
        write_root(T.blend_transform(read_root(a), read_root(b), w), want)
        assert np.array_equal(dof_token.poses[k], want)


def test_sampling_at_exact_frame_times():
    poses = np.zeros((31, 36), np.float32)
    poses[:, 6] = 1.0
    poses[:, 10] = np.arange(31)
    buffer = rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, 30.0, poses, foot_contacts=np.zeros((31, 4), np.float32))
    for k in range(31):
        assert buffer.sample(k / 30.0)[10] == pytest.approx(k, abs=1e-5)
    assert rt.frame_index_from_time(buffer.duration, 30.0, 31) == (30, 30, 0.0)
    assert buffer.sample(buffer.duration)[10] == 30.0 and buffer.sample(buffer.duration + 1e-9)[10] == 30.0
    assert rt.frame_index_from_time(0.0, 30.0, 0) == (0, 0, 0.0)


def test_settings_are_validated():
    for bad in ({"transition_early_by_frames": -1, "num_poses_to_use": 4}, {"transition_early_by_frames": 0, "num_poses_to_use": 0},
                {"transition_early_by_frames": 0, "num_poses_to_use": 5}, {"transition_early_by_frames": 2.5, "num_poses_to_use": 4},
                {"transition_early_by_frames": 0, "num_poses_to_use": 4, "ik_pin_mask": 16},
                {"transition_early_by_frames": 0, "num_poses_to_use": 4, "allowed_token_counts": (8.0,)}):
        with pytest.raises(ValueError):
            rt.TargetPoseInferenceParams(**bad)
    params = rt.TargetPoseInferenceParams(np.int64(3), np.int32(2), ik_pin_mask=5, allowed_token_counts=[np.int64(9)])
    assert params.transition_early_by_frames == 3 and type(params.num_poses_to_use) is int
    assert params.ik_pin_mask == rt.IKPin.LEFT_HAND | rt.IKPin.LEFT_FOOT and params.allowed_token_counts == (9,)
    with pytest.raises(ValueError, match="integers"):
        rt.MotionBricksInferenceConfig(allowed_token_counts=(8.0,))
    assert rt.MotionBricksInferenceConfig(allowed_token_counts=[np.int64(7)]).allowed_token_counts == (7,)


def test_shapes_are_checked_at_the_api_boundary():
    buffer = rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, 30.0, np.tile(T.IDENTITY.tolist() + [0.0] * 29, (3, 1)),
                                   foot_contacts=np.eye(3, 4, dtype=np.float32))
    with pytest.raises(ValueError, match="foot contacts"):
        buffer.sample_foot_contacts(0.0, np.zeros((2, 4), np.float32))
    with pytest.raises(ValueError, match="coordinates"):
        buffer.sample(0.0, np.zeros(35, np.float32))
    pose = np.tile(T.IDENTITY, (30, 1))
    with pytest.raises(ValueError, match="pose\\[None\\]"):
        rt.place_target_poses([1.0, 2.0, 0.0], [0.0, 1.0, 0.0], pose)
    with pytest.raises(ValueError, match="shape of the poses"):
        rt.place_target_poses([1.0, 2.0, 0.0], [0.0, 1.0, 0.0], pose[None], np.zeros((2, 30, 7), np.float32))
    placed = rt.place_target_dof([1.0, 2.0, 0.0], [0.0, 1.0, 0.0], buffer.poses[0])  # one pose of coordinates works
    assert placed.shape == (36,) and placed[0] == pytest.approx(1.0)
    reference = np.zeros(36, np.float32)
    reference[6] = 1.0
    player = rt.GeneratedDOFMotionPlayer(_FakeEngine(31), reference)
    with pytest.raises(ValueError, match="pose\\[None\\]"):
        player.generate(reference, rt.TargetPoseInferenceParams(0, 1))


def test_public_namespace_and_constants():
    namespace = {}
    exec("import io\nfrom motionbricks_mlx.runtime import *", namespace)
    assert namespace["io"].__name__ == "io" and "engine" not in namespace and "transforms" in namespace
    assert all(hasattr(rt, name) for name in rt.__all__) and len(set(rt.__all__)) == len(rt.__all__)
    for constant in (T.IDENTITY, T.IDENTITY_QUAT):
        with pytest.raises(ValueError):
            constant[0] = 1.0


def test_csv_round_trip(tmp_path):
    poses = np.concatenate([RNG.normal(size=(12, 3)), random_quat(12), RNG.uniform(-3, 3, size=(12, 29))], -1)
    buffer = rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, 30.0, poses, num_frames=10)
    rt.save_csv(tmp_path / "motion.csv", buffer)
    loaded = rt.load_csv(tmp_path / "motion.csv", sample_rate=30.0)  # the file does not store the rate
    assert loaded.num_frames == 10 and loaded.joint_names == buffer.joint_names
    same_dof(loaded.poses, buffer.poses[:10], 1e-5)


@pytest.mark.parametrize("rate", [0.0, -30.0, float("nan"), float("inf")])
def test_sample_rate_must_be_positive_and_finite(rate, tmp_path):
    poses = np.concatenate([np.zeros((2, 3)), np.tile([0, 0, 0, 1.0], (2, 1)), np.zeros((2, 29))], -1)
    with pytest.raises(ValueError, match="sample_rate"):
        rt.DOFAnimationBuffer(rt.UNITREE_G1_JOINT_NAMES, rate, poses)
    bvh = tmp_path / "bad.bvh"
    frame_time = "0" if rate == 0.0 else str(rate)
    bvh.write_text("HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
                   "  End Site\n  {\n    OFFSET 0 1 0\n  }\n}\nMOTION\nFrames: 1\n"
                   f"Frame Time: {frame_time}\n0 0 0 0 0 0\n")
    with pytest.raises(ValueError, match="frame time"):
        rt.load_bvh(bvh)


# --- teacher-forced parity with upstream's runtime ----------------------------------------------


def golden(name):
    path = RUNTIME_GOLDEN / f"{name}.npz"
    if not path.is_file():
        pytest.skip(f"runtime fixture {path} missing (made by the sandbox: sandbox/scripts/make_runtime_golden.py)")
    return dict(np.load(path))


def joint_names(g):
    return bytes(g["skeleton.names_utf8"].astype(np.uint8)).decode().split("\n")


def soma_skeleton(g):
    return rt.Skeleton(joint_names(g), g["skeleton.parents"], g["skeleton.reference"])


def engine_for(name, g, label, model=None):
    m2r = g[f"gen.{label}.model_to_runtime"]
    cls = rt.MotionBricksSOMAEngine if name == "soma" else rt.MotionBricksUnitreeG1Engine
    return cls(model, model_to_runtime=m2r)


def tokens_for(name, g, label, engine):
    args = (soma_skeleton(g),) if name == "soma" else ()
    context, target, output = engine.create_input(*args), engine.create_input(*args), engine.create_output(*args)
    context.update_from_poses(g[f"gen.{label}.context"])
    target.update_from_poses(g[f"gen.{label}.target"])
    config = rt.MotionBricksInferenceConfig(allowed_token_counts=tuple(int(v) for v in g[f"gen.{label}.allowed"]),
                                            heading_alignment=tuple(float(v) for v in g[f"gen.{label}.heading"]),
                                            correct_generated_diff=bool(g[f"gen.{label}.correct"]))
    return context, target, output, config


@pytest.mark.parametrize("label", SCENARIOS)
def test_encode_matches_upstream(model_name, label):
    g = golden(model_name)
    engine = engine_for(model_name, g, label)
    context, target, output, config = tokens_for(model_name, g, label, engine)
    engine._prepare_generation(context, target, output)
    inputs = engine.encode_inputs(context, target, config)
    same_transforms(engine.alignment, g[f"gen.{label}.alignment"], 1e-5)
    for key, value in inputs.items():
        want = g[f"gen.{label}.in.{key}"]
        assert value.shape == want.shape and value.dtype == want.dtype, key
        if model_name == "unitree_g1" and key in ("context_qpos", "target_qpos"):
            same_dof(value[0], want[0], 1e-5)
        else:
            np.testing.assert_allclose(value, want, atol=2e-5, err_msg=key)


@pytest.mark.parametrize("label", SCENARIOS)
def test_decode_and_endpoint_correction_match_upstream(model_name, label):
    g = golden(model_name)
    engine = engine_for(model_name, g, label)
    context, target, output, config = tokens_for(model_name, g, label, engine)
    engine._prepare_generation(context, target, output)
    engine.encode_inputs(context, target, config)  # sets the alignment
    prediction = {k[len(f"gen.{label}.out."):]: v for k, v in g.items() if k.startswith(f"gen.{label}.out.")}
    count = engine.decode_prediction(prediction, output)
    assert count == int(g[f"gen.{label}.frame_count"])
    check = same_transforms if model_name == "soma" else same_dof
    check(output.poses[:count], g[f"gen.{label}.decoded"], 2e-5)
    np.testing.assert_allclose(output.foot_contacts[:count], g[f"gen.{label}.decoded_contacts"], atol=1e-6)
    if config.correct_generated_diff:
        engine.correct_endpoints(context, target, output)
    check(output.poses[:count], g[f"gen.{label}.generated"], 2e-5)


def test_clip_loading_matches_upstream(model_name):
    g = golden(model_name)
    names = {"walk": "Neutral_walk_forward_002__A057", "jog": "neutral_jog_ff_180_R_001__A534"}
    for label, stem in names.items():
        path = CLIPS / model_name / "locomotion" / "bs_1d_linear" / f"{stem}.{'bvh' if model_name == 'soma' else 'csv'}"
        if not path.is_file() or path.stat().st_size < 1024:
            pytest.skip(f"clip {path} missing (motionbricks-mlx download --clips, or the sandbox)")
        clip = rt.load_bvh(path, soma_skeleton(g)) if model_name == "soma" else rt.load_csv(path)
        assert clip.num_frames == int(g[f"clip.{label}.num_frames"]) and clip.sample_rate == float(g[f"clip.{label}.sample_rate"])
        frames = clip.poses[g[f"clip.{label}.frame_index"]]
        (same_transforms if model_name == "soma" else same_dof)(frames, g[f"clip.{label}.frames"], 2e-5)
    if model_name == "soma":
        skeleton = rt.load_skeleton_from_bvh(CLIPS / "soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh", remove_root=True)
        assert skeleton.joint_names == joint_names(g) and np.array_equal(skeleton.parent_indices, g["skeleton.parents"])
        same_transforms(skeleton.reference_local_transforms, g["skeleton.reference"], 1e-6)


def test_recorder_sampling_matches_upstream(model_name):
    g = golden(model_name)
    poses = g["recorder.poses"]
    history = rt.MotionRecorder(poses[0], capacity=8)
    for dt, pose in zip(g["recorder.dts"], poses[1:]):
        history.record(float(dt), pose)
    np.testing.assert_allclose(history.delta_times, g["recorder.delta_times"], atol=0)
    assert np.array_equal(history.poses, g["recorder.history"])
    token = rt.PoseToken(poses.shape[1]) if model_name == "soma" else rt.DOFToken(29)
    token.update_from_recorder(history)
    (same_transforms if model_name == "soma" else same_dof)(token.poses, g["recorder.token"], 2e-6)


def test_target_placement_and_speed_match_upstream(model_name):
    g = golden(model_name)
    targets = g["target.poses"]
    soma = model_name == "soma"
    place = rt.place_target_poses if soma else rt.place_target_dof
    check = same_transforms if soma else same_dof
    check(place(np.array([3.0, -1.5, 0.0]), np.array([0.3, 0.9, 0.0]), targets), g["target.placed"], 2e-6)
    for speed in (0.0, 1.3):
        scaled = targets.copy()
        (rt.adjust_target_pose_speed if soma else rt.adjust_target_dof_speed)(scaled, speed)
        check(scaled, g[f"target.speed_{speed:g}"], 2e-6)


def test_buffer_sampling_and_looping_match_upstream(model_name):
    g = golden(model_name)
    names = {"soma": "Neutral_walk_forward_002__A057.bvh", "unitree_g1": "Neutral_walk_forward_002__A057.csv"}
    path = CLIPS / model_name / "locomotion" / "bs_1d_linear" / names[model_name]
    if not path.is_file() or path.stat().st_size < 1024:
        pytest.skip(f"clip {path} missing")
    clip = rt.load_bvh(path, soma_skeleton(g)) if model_name == "soma" else rt.load_csv(path)
    check = same_transforms if model_name == "soma" else same_dof
    offset = T.transform([1.0, 2.0, 0.0], T.quat_from_axis_angle([0.0, 0.0, 1.0], np.float32(np.radians(40.0))))
    for t in (0.0, 0.37, 1.234, 99.0):
        check(clip.sample(t, root_offset=offset), g[f"sample.{t:g}"], 2e-5)
    loop = clip.copy()
    loop.poses[:40] = g["loop.before"]
    loop.num_frames = 40
    loop.make_loopable()
    check(loop.poses[:40], g["loop.after"], 2e-5)


def test_ik_pins_match_upstream():
    g = golden("soma")
    skeleton = soma_skeleton(g)
    corrector = rt.TargetPoseIKCorrector(skeleton, rt.create_soma_target_pose_ik_rig(skeleton))
    for label in ("hands", "all"):
        corrector.capture_pins(g["ik.target"], rt.IKPin(int(g[f"ik.{label}.mask"])))
        pose = g["ik.current"].copy()
        corrector.apply(pose, float(g[f"ik.{label}.weight"]))
        same_transforms(pose, g[f"ik.{label}.result"], 5e-5)
    corrector.capture_pins(g["ik.target"], rt.IKPin.NONE)
    pose = g["ik.current"].copy()
    corrector.apply(pose, 1.0)
    assert np.array_equal(pose, g["ik.current"])


class _RecordedModel:
    """A backend that returns upstream's recorded prediction (any ``run(inputs)`` backend works)."""

    def __init__(self, g, label, num_tokens=None):
        prefix = f"gen.{label}.out."
        self.outputs = {k[len(prefix):]: v for k, v in g.items() if k.startswith(prefix)}
        if num_tokens is not None:
            self.outputs["pred_num_tokens"] = np.array([num_tokens], np.int32)
        self.inputs = None

    def run(self, inputs):
        self.inputs = inputs
        return self.outputs


@pytest.mark.parametrize("label", SCENARIOS)
def test_generate_animation_with_a_recorded_backend(model_name, label):
    """The whole ``generate_animation()`` without weights: stages, frame count, settings and timer."""
    g = golden(model_name)
    backend = _RecordedModel(g, label)
    engine = engine_for(model_name, g, label, backend)
    context, target, output, config = tokens_for(model_name, g, label, engine)
    assert engine.generate_animation(context, target, output, config=config) is output
    count = int(g[f"gen.{label}.frame_count"])
    assert output.num_frames == count and set(backend.inputs) == {k[len(f"gen.{label}.in."):] for k in g if k.startswith(f"gen.{label}.in.")}
    (same_transforms if model_name == "soma" else same_dof)(output.poses[:count], g[f"gen.{label}.generated"], 2e-5)
    np.testing.assert_allclose(output.foot_contacts[:count], g[f"gen.{label}.decoded_contacts"], atol=1e-6)
    assert set(engine.timer.milliseconds) == {"encode", "inference", "decode", "post_process"}
    assert (engine.timer.milliseconds["post_process"] == 0.0) == (not config.correct_generated_diff)
    outside = min(set(range(6, 17)) - set(config.allowed_token_counts))
    with pytest.raises(RuntimeError, match="outside the allowed set"):
        engine_for(model_name, g, label, _RecordedModel(g, label, outside)).generate_animation(context, target, output, config=config)


def test_soma_decode_matches_joint_by_joint():
    g = golden("soma")
    engine = engine_for("soma", g, "walk")
    context, target, output, config = tokens_for("soma", g, "walk", engine)
    engine._prepare_generation(context, target, output)
    engine.encode_inputs(context, target, config)
    count = engine.decode_prediction(_RecordedModel(g, "walk").outputs, output)
    from motionbricks_mlx.runtime.engine import _rotation_from_6d

    skeleton, jm = output.skeleton, len(rt.SOMA_JOINT_NAMES)
    raw = g["gen.walk.out.pred_global_motions"][0, 3: 3 + count]
    six = raw[:, 5 + (jm - 1) * 3: 5 + (jm - 1) * 3 + 6 * jm].reshape(count, jm, 6)
    rotations = _rotation_from_6d(six[..., :3], six[..., 3:])
    pose_to_model, ref = engine._pose_to_model, skeleton.reference_local_transforms
    assert (pose_to_model < 0).sum() > 0  # the fixture skeleton has joints that keep their reference
    for joint in range(1, skeleton.num_joints):
        if pose_to_model[joint] < 0:
            want = np.broadcast_to(ref[joint], (count, 7))
        else:
            parent = pose_to_model[skeleton.parent_indices[joint]]
            local = T.quat_mul(T.quat_inverse(rotations[:, parent]), rotations[:, pose_to_model[joint]])
            want = T.transform(np.broadcast_to(ref[joint, :3], (count, 3)), T.quat_normalize(local))
        assert np.array_equal(output.poses[:count, joint], want), joint


def test_soma_player_with_a_recorded_backend():
    g = golden("soma")
    engine = engine_for("soma", g, "walk", _RecordedModel(g, "walk"))
    skeleton = soma_skeleton(g)
    player = rt.GeneratedMotionPlayer(engine, skeleton)
    player.seed_history_from_pose(g["gen.walk.context"][-1])
    player.generate(g["gen.walk.target"], rt.TargetPoseInferenceParams(transition_early_by_frames=4, num_poses_to_use=4))
    count = int(g["gen.walk.frame_count"])
    assert player.output.num_frames == count and player.time_to_transition == pytest.approx((count - 1 - 4) / 30)
    pose, contacts = np.zeros((skeleton.num_joints, 7), np.float32), np.zeros(4, np.float32)
    for _ in range(5):
        player.sample(1 / 60, pose, contacts)
        player.record_pose(1 / 60, pose)
    assert np.array_equal(pose, player.output.sample(player.playback_time)) and np.array_equal(player.motion_history.poses[-1], pose)
    np.testing.assert_allclose(player.current_root_position(), pose[0, :3], atol=1e-6)


def test_soma_engine_rejects_unusable_skeletons():
    g = golden("soma")
    names, parents, reference = joint_names(g), g["skeleton.parents"], g["skeleton.reference"]
    engine = rt.MotionBricksSOMAEngine(None)
    renamed = [n if n != "LeftShin" else "LeftKnee" for n in names]
    with pytest.raises(ValueError, match="LeftShin"):
        engine.create_output(rt.Skeleton(renamed, parents, reference))
    rooted = rt.Skeleton(["Root", *names], [-1, *(p + 1 for p in parents)], np.concatenate([T.IDENTITY[None], reference]))
    with pytest.raises(ValueError, match="index zero"):
        engine.create_input(rooted)


# --- end to end with the MLX model -------------------------------------------------------------


@pytest.mark.weights
@pytest.mark.parametrize("label", SCENARIOS)
def test_generation_matches_upstream(model_name, label):
    g = golden(model_name)
    engine = engine_for(model_name, g, label, load_model(model_name))
    context, target, output, config = tokens_for(model_name, g, label, engine)
    engine.generate_animation(context, target, output, config=config)
    count = int(g[f"gen.{label}.frame_count"])
    assert output.num_frames == count, "durations differ from upstream"
    # The MLX model matches PyTorch to ~1e-5; decoded rotations pass a Gram-Schmidt step.
    (same_transforms if model_name == "soma" else same_dof)(output.poses[:count], g[f"gen.{label}.generated"], 3e-3)
    np.testing.assert_allclose(output.foot_contacts[:count], g[f"gen.{label}.decoded_contacts"], atol=5e-3)
    assert set(engine.timer.milliseconds) == {"encode", "inference", "decode", "post_process"}
