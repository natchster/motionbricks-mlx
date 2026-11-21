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
