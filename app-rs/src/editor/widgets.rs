// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The imgui widgets shared by the editors, ported from upstream
//! `app/authoring/common.py`: inline rename, outliner tree rows, drag-and-drop,
//! labeled property rows, the token-range widget, and the transform editor with
//! reset menus and per-axis locks.
//!
//! Where upstream calls the Python `imgui` module directly, these functions take
//! `&imgui::Ui` (imgui-rs 0.12). The pure math (`stable_rotation_degrees`,
//! `apply_transform_locks`, `transforms_equal`, angle wrapping) is kept
//! UI-free and unit-tested below.

use std::collections::HashSet;
use std::path::Path;

use glam::{Quat, Vec3};
use imgui::{Condition, DragDropFlags, DragRange, ItemHoveredFlags, Key, MouseButton, SliderFlags, StyleColor, TreeNodeFlags, TreeNodeToken, Ui};

use crate::documents::smart_primitive::{MODEL_MAX_TOKEN_COUNT, MODEL_MIN_TOKEN_COUNT, MODEL_POSES_PER_TOKEN};
use crate::documents::{ModelRig, Transform};

/// Upstream `WINDOW_MARGIN`: gap between the viewport edge and the panels.
pub const WINDOW_MARGIN: f32 = 10.0;
/// Upstream `RIGHT_DOCK_RATIO`: width of the docked right column as a fraction of the window.
pub const RIGHT_DOCK_RATIO: f32 = 0.30;
/// Upstream `PROPERTIES_WINDOW_WIDTH` / `PROPERTIES_WINDOW_HEIGHT`.
pub const PROPERTIES_WINDOW_WIDTH: f32 = 440.0;
pub const PROPERTIES_WINDOW_HEIGHT: f32 = 420.0;
/// Upstream `_PROPERTIES_DOCK_RATIO`: the Properties panel's share of the right column.
pub const PROPERTIES_DOCK_RATIO: f32 = 0.52;
/// Upstream `NAME_TEXT_COLOR`: the light blue of item names.
pub const NAME_TEXT_COLOR: [f32; 4] = [0.60, 0.80, 1.0, 1.0];
/// Upstream `_MENU_BAR_ALPHA`.
pub const MENU_BAR_ALPHA: f32 = 0.8;
/// Upstream `_PANEL_ALPHA`.
pub const PANEL_ALPHA: f32 = 0.9;
/// Upstream `_LOCKED_AXIS_COLOR`: locked axis labels draw red.
pub const LOCKED_AXIS_COLOR: [f32; 4] = [1.0, 0.3, 0.3, 1.0];
/// Upstream `_HIDDEN_NAME_ALPHA`: hidden items draw dimmed in the tree.
pub const HIDDEN_NAME_ALPHA: f32 = 0.45;
/// Upstream `_DISPLAY_ZERO_EPSILON`: angles smaller than this display as 0.
pub const DISPLAY_ZERO_EPSILON: f32 = 5.0e-4;
/// Upstream `POSES_TO_USE_TOOLTIP`.
pub const POSES_TO_USE_TOOLTIP: &str = "How many of the target's four poses inform the model. \
    One pose makes the motion stop at the target. More poses add velocity, acceleration, and momentum.";

const FULL_ROTATION: f32 = 360.0;
const HALF_ROTATION: f32 = 0.5 * FULL_ROTATION;
const AXIS_NAMES: [&str; 3] = ["X", "Y", "Z"];

/// Upstream `RIG_LABELS` (`app/runtime/common/rig.py`).
pub const RIG_LABELS: [(ModelRig, &str); 2] = [(ModelRig::Soma, "SOMA"), (ModelRig::UnitreeG1, "Unitree G1")];

/// File menu commands that an editor can request (upstream `FileAction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileAction {
    New,
    Open,
    Save,
    SaveAs,
    Exit,
}

/// Rename state for one item, drawn as a text field in place of its name
/// (upstream `InlineRename`).
#[derive(Debug, Default)]
pub struct InlineRename {
    pub item_id: Option<u32>,
    pub text: String,
    focus: bool,
}

impl InlineRename {
    /// Start renaming an item, with its current name in the field.
    pub fn begin(&mut self, item_id: u32, name: &str) {
        self.item_id = Some(item_id);
        self.text = name.to_string();
        self.focus = true;
    }

    /// Draw the rename field, and return the new name once it is confirmed.
    ///
    /// Enter or leaving the field confirms it. Escape cancels it.
    pub fn draw(&mut self, ui: &Ui) -> Option<String> {
        ui.set_next_item_width(-1.0);
        if self.focus {
            ui.set_keyboard_focus_here();
            self.focus = false;
        }
        let submitted = ui
            .input_text(format!("##rename_{}", self.item_id.unwrap_or(0)), &mut self.text)
            .enter_returns_true(true)
            .auto_select_all(true)
            .build();
        if ui.is_key_pressed(Key::Escape) {
            self.item_id = None;
        } else if submitted || ui.is_item_deactivated() {
            self.item_id = None;
            let trimmed = self.text.trim();
            return (!trimmed.is_empty()).then(|| trimmed.to_string());
        }
        None
    }
}

/// Draw a label, then place and size the next widget on the same row (upstream
/// `draw_label`). `column` is where the widget starts in pixels from the
/// window's left edge; zero places it right after the label. `width` is the
/// widget width in pixels; -1 fills the rest of the row.
pub fn draw_label(ui: &Ui, label: &str, tooltip: Option<&str>, column: f32, width: f32) {
    ui.align_text_to_frame_padding();
    ui.text(format!("{label}:"));
    if let Some(tooltip) = tooltip {
        // Upstream shows the tooltip after a delay; imgui-rs shows it on hover.
        if ui.is_item_hovered() {
            ui.tooltip_text(tooltip);
        }
    }
    if column > 0.0 {
        ui.same_line_with_pos(column);
    } else {
        ui.same_line();
    }
    ui.set_next_item_width(width);
}

/// Draw the + and - buttons of a panel, and return whether each one was clicked
/// (upstream `draw_add_remove_buttons`).
pub fn draw_add_remove_buttons(ui: &Ui, can_remove: bool) -> (bool, bool) {
    let add = ui.button("+");
    ui.same_line();
    let remove = {
        let _disabled = ui.begin_disabled(!can_remove);
        ui.button("-")
    };
    (add, remove)
}

/// The outcome of [`draw_tree_row`].
///
/// When `token` is `Some` the row is open: the caller draws the row's children
/// and then drops the token (upstream calls `ui.tree_pop()`).
pub struct TreeRow<'ui> {
    pub token: Option<TreeNodeToken<'ui>>,
    /// The user selected the row this frame.
    pub clicked: bool,
    /// The confirmed new name, once a rename completes.
    pub new_name: Option<String>,
}

/// Draw one row of an outliner tree (upstream `draw_tree_row`).
///
/// A click selects the row. A double click selects it and starts renaming it.
/// `reveal_ids` holds the rows to open so a new selection shows; the selected
/// row scrolls into view and clears the set.
#[allow(clippy::too_many_arguments)]
pub fn draw_tree_row<'ui>(
    ui: &'ui Ui,
    name: &str,
    item_id: u32,
    rename: &mut InlineRename,
    reveal_ids: &mut HashSet<u32>,
    selected: bool,
    leaf: bool,
    renamable: bool,
    hidden: bool,
) -> TreeRow<'ui> {
    let renaming = rename.item_id == Some(item_id);
    let mut flags = TreeNodeFlags::OPEN_ON_ARROW | TreeNodeFlags::DEFAULT_OPEN;
    if !renaming {
        flags |= TreeNodeFlags::SPAN_AVAIL_WIDTH;
    }
    if leaf {
        flags |= TreeNodeFlags::LEAF;
    }
    if selected {
        flags |= TreeNodeFlags::SELECTED;
    }
    let reveal = reveal_ids.contains(&item_id);
    let label = if renaming { "###TreeRow".to_string() } else { format!("{name}###TreeRow") };
    let mut node = ui.tree_node_config(&label).flags(flags);
    if reveal && !selected {
        node = node.opened(true, Condition::Always);
    }
    let text_color = if hidden {
        [NAME_TEXT_COLOR[0], NAME_TEXT_COLOR[1], NAME_TEXT_COLOR[2], HIDDEN_NAME_ALPHA]
    } else {
        NAME_TEXT_COLOR
    };
    let token = {
        let _color = ui.push_style_color(StyleColor::Text, text_color);
        node.push()
    };
    if reveal && selected {
        ui.set_scroll_here_y();
        reveal_ids.clear();
    }

    if renaming {
        ui.same_line();
        let new_name = rename.draw(ui);
        return TreeRow { token, clicked: false, new_name };
    }
    let toggled = ui.is_item_toggled_open();
    if renamable && ui.is_item_hovered() && ui.is_mouse_double_clicked(MouseButton::Left) && !toggled {
        rename.begin(item_id, name);
        return TreeRow { token, clicked: true, new_name: None };
    }
    TreeRow { token, clicked: ui.is_item_clicked() && !toggled, new_name: None }
}

/// Let the last widget be dragged with `item_id` as its payload (upstream
/// `drag_source`). `text` follows the pointer during the drag. Returns true
/// while the widget is dragged.
pub fn drag_source(ui: &Ui, payload_type: &str, item_id: u32, text: &str) -> bool {
    match ui.drag_drop_source_config(payload_type).begin_payload(item_id) {
        Some(_tooltip) => {
            ui.text(text);
            true
        }
        None => false,
    }
}

/// Return the item ID dropped on the last widget in this frame, or `None`
/// (upstream `accept_drop`).
pub fn accept_drop(ui: &Ui, payload_type: &str) -> Option<u32> {
    let target = ui.drag_drop_target()?;
    let dropped = match target.accept_payload::<u32, _>(payload_type, DragDropFlags::empty()) {
        Some(Ok(payload)) if payload.delivery => Some(payload.data),
        _ => None,
    };
    target.pop();
    dropped
}

/// Draw a labeled integer slider that fills the available width (upstream
/// `draw_int_slider_property`).
pub fn draw_int_slider_property(
    ui: &Ui,
    label: &str,
    value: i32,
    minimum: i32,
    maximum: i32,
    item_id: u32,
    tooltip: Option<&str>,
) -> i32 {
    draw_label(ui, label, tooltip, 0.0, -1.0);
    let mut value = value;
    ui.slider_config(format!("##{label}{item_id}"), minimum, maximum)
        .flags(SliderFlags::ALWAYS_CLAMP)
        .build(&mut value);
    value
}

/// Draw the Tokens to Generate range and its raw frames, then return the range
/// (upstream `draw_token_range_property`).
pub fn draw_token_range_property(ui: &Ui, min_tokens: u32, max_tokens: u32, item_id: u32) -> (u32, u32) {
    draw_label(ui, "Tokens to Generate", None, 0.0, -1.0);
    let (mut min_tokens, mut max_tokens) = (min_tokens as i32, max_tokens as i32);
    DragRange::new(format!("##tokens{item_id}"))
        .range(MODEL_MIN_TOKEN_COUNT as i32, MODEL_MAX_TOKEN_COUNT as i32)
        .speed(1.0)
        .display_format("Min: %d")
        .max_display_format("Max: %d")
        .flags(SliderFlags::ALWAYS_CLAMP)
        .build(ui, &mut min_tokens, &mut max_tokens);
    let min_frames = min_tokens as u32 * MODEL_POSES_PER_TOKEN;
    let max_frames = max_tokens as u32 * MODEL_POSES_PER_TOKEN;
    let frame_range = if min_frames == max_frames {
        min_frames.to_string()
    } else {
        format!("{min_frames} to {max_frames}")
    };
    ui.text(format!("Raw frames: {frame_range}"));
    (min_tokens as u32, max_tokens as u32)
}

/// Draw Width, Depth, and Height fields in meters (upstream
/// `draw_box_dimensions`). Edited values are at least `minimum`; a value that
/// is not finite becomes `minimum`.
pub fn draw_box_dimensions(ui: &Ui, dimensions: [f32; 3], minimum: f32, dpi_scale: f32) -> (bool, [f32; 3]) {
    let mut values = dimensions;
    let mut changed = false;
    for (index, label) in ["Width", "Depth", "Height"].iter().enumerate() {
        draw_label(ui, label, None, 80.0 * dpi_scale, 100.0 * dpi_scale);
        changed |= ui.input_float(format!("##{label}"), &mut values[index]).build();
    }
    if changed {
        for value in &mut values {
            *value = if value.is_finite() { value.max(minimum) } else { minimum };
        }
    }
    (changed, values)
}

/// Return 0 for a value too small to show (upstream `clean_display_value`).
pub fn clean_display_value(value: f32) -> f32 {
    if value.abs() < DISPLAY_ZERO_EPSILON { 0.0 } else { value }
}

/// The frame nearest `sample_time`, clamped to the clip (upstream
/// `draw_clip_scrubber`'s frame math).
pub fn sample_frame_at(sample_time: f32, sample_rate: f32, last_frame: u32) -> u32 {
    ((sample_time * sample_rate).round() as i64).clamp(0, last_frame as i64) as u32
}

/// Clamp a token range to the model contract, with the maximum never below
/// the minimum (upstream `TargetPoseComponent.set_token_range`).
pub fn clamp_token_range(min_tokens: u32, max_tokens: u32) -> (u32, u32) {
    let min_tokens = min_tokens.clamp(MODEL_MIN_TOKEN_COUNT, MODEL_MAX_TOKEN_COUNT);
    let max_tokens = max_tokens.clamp(min_tokens, MODEL_MAX_TOKEN_COUNT);
    (min_tokens, max_tokens)
}

/// Roll, pitch, yaw of `rotation` in radians, upstream's `wp.quat_to_rpy()`
/// (same formulas as [`Transform::euler_degrees`], without the cleaning).
pub fn quat_to_rpy(rotation: Quat) -> Vec3 {
    let (x, y, z, w) = (rotation.x, rotation.y, rotation.z, rotation.w);
    let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
    let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
    let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
    Vec3::new(roll, pitch, yaw)
}

/// Keep displayed rotation angles close to their previous values to avoid
/// jumps (upstream `stable_rotation_degrees`). `rotation` is the authoritative
/// quaternion; `reference` holds the angles shown last frame, in degrees.
pub fn stable_rotation_degrees(rotation: Quat, reference: Option<Vec3>) -> Vec3 {
    let rpy = quat_to_rpy(rotation);
    let mut primary = Vec3::new(rpy.x.to_degrees(), rpy.y.to_degrees(), rpy.z.to_degrees());
    if let Some(reference) = reference {
        let mut nearest = Vec3::ZERO;
        let mut alternate = Vec3::ZERO;
        for axis in 0..3 {
            nearest[axis] = nearest_angle(primary[axis], reference[axis]);
        }
        alternate[0] = nearest_angle(primary[0] + HALF_ROTATION, reference[0]);
        alternate[1] = nearest_angle(HALF_ROTATION - primary[1], reference[1]);
        alternate[2] = nearest_angle(primary[2] + HALF_ROTATION, reference[2]);
        let primary_distance = (nearest - reference).length_squared();
        let alternate_distance = (alternate - reference).length_squared();
        primary = if alternate_distance < primary_distance { alternate } else { nearest };
    }
    Vec3::new(
        clean_display_value(primary.x),
        clean_display_value(primary.y),
        clean_display_value(primary.z),
    )
}

/// Compare two transforms, ignoring tiny differences and treating q and -q as
/// the same rotation (upstream `transforms_equal`).
pub fn transforms_equal(lhs: &Transform, rhs: &Transform) -> bool {
    let position_equal = lhs.translation.abs_diff_eq(rhs.translation, 1e-6);
    let rotation_equal = lhs.rotation_xyzw.abs_diff_eq(rhs.rotation_xyzw, 1e-6)
        || lhs.rotation_xyzw.abs_diff_eq(-rhs.rotation_xyzw, 1e-6);
    position_equal && rotation_equal
}

/// Shorten text with ... to fit `max_width` in pixels (upstream `ellipsize_text`).
pub fn ellipsize_text(ui: &Ui, text: &str, max_width: f32) -> String {
    if max_width <= 0.0 {
        return String::new();
    }
    if ui.calc_text_size(text)[0] <= max_width {
        return text.to_string();
    }
    let ellipsis = "...";
    if ui.calc_text_size(ellipsis)[0] > max_width {
        return String::new();
    }
    // Binary search over the character count (not bytes, so UTF-8 stays intact).
    let (mut lower, mut upper) = (0usize, text.chars().count());
    while lower < upper {
        let middle = (lower + upper + 1) / 2;
        let candidate: String = text.chars().take(middle).collect();
        if ui.calc_text_size(format!("{candidate}{ellipsis}"))[0] <= max_width {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    let prefix: String = text.chars().take(lower).collect();
    format!("{prefix}{ellipsis}")
}

/// Show the Character combo and return the chosen rig (upstream
/// `draw_character_selector`). When `locked_reason` is given, the combo is
/// disabled and its tooltip shows the reason.
pub fn draw_character_selector(ui: &Ui, rig: ModelRig, locked_reason: Option<&str>) -> ModelRig {
    let labels: Vec<&str> = RIG_LABELS.iter().map(|(_, label)| *label).collect();
    let mut index = RIG_LABELS.iter().position(|(candidate, _)| *candidate == rig).unwrap_or(0);
    draw_label(ui, "Character", None, 0.0, -1.0);
    {
        let _disabled = ui.begin_disabled(locked_reason.is_some());
        ui.combo_simple_string("##Character", &mut index, &labels);
    }
    if let Some(reason) = locked_reason {
        if ui.is_item_hovered_with_flags(ItemHoveredFlags::ALLOW_WHEN_DISABLED) {
            ui.tooltip_text(reason);
        }
    }
    RIG_LABELS[index].0
}

/// Show a file name after the label, with its path in a tooltip; return true
/// when the load button is clicked (upstream `draw_clip_selector`).
pub fn draw_clip_selector(ui: &Ui, name: &str, path: Option<&Path>, item_id: u32, label: &str) -> bool {
    let (frame_padding_x, item_spacing_x) = unsafe {
        let style = &*imgui::sys::igGetStyle();
        (style.FramePadding.x, style.ItemSpacing.x)
    };
    let button_width = ui.calc_text_size("...")[0] + 2.0 * frame_padding_x;
    ui.align_text_to_frame_padding();
    ui.text(format!("{label}:"));
    ui.same_line();
    let name_width = ui.content_region_avail()[0] - button_width - item_spacing_x;
    ui.text(ellipsize_text(ui, name, name_width));
    if let Some(path) = path {
        if ui.is_item_hovered() {
            ui.tooltip_text(path.display().to_string());
        }
    }
    ui.same_line();
    ui.button_with_size(format!("...##LoadClip{item_id}"), [button_width, 0.0])
}

/// The outcome of [`draw_transform_editor`].
pub struct TransformEdit {
    /// Whether a value changed.
    pub changed: bool,
    /// The edited transform.
    pub transform: Transform,
    /// The angles to pass back on the next call.
    pub euler_degrees: Vec3,
}

/// Edit position and rotation, with reset menus and axis locks (upstream
/// `draw_transform_editor`). `euler_degrees` holds the angles from the previous
/// call (they keep the shown angles from jumping); `None` on the first call.
/// The lock arrays are edited in place by the lock menus.
pub fn draw_transform_editor(
    ui: &Ui,
    transform: &Transform,
    euler_degrees: Option<Vec3>,
    scale: f32,
    translation_locks: &mut [bool; 3],
    rotation_locks: &mut [bool; 3],
) -> TransformEdit {
    let mut position = [
        clean_display_value(transform.translation.x),
        clean_display_value(transform.translation.y),
        clean_display_value(transform.translation.z),
    ];
    let mut rotation = stable_rotation_degrees(transform.rotation_xyzw, euler_degrees).to_array();
    let moved = draw_transform_row(ui, "Translation", &mut position, scale, translation_locks);
    let rotated = draw_transform_row(ui, "Rotation", &mut rotation, scale, rotation_locks);
    let original = transform.translation.to_array();
    for axis in 0..3 {
        if !(moved && !translation_locks[axis]) {
            position[axis] = original[axis];
        }
    }
    let quaternion = if rotated {
        Transform::quat_from_euler_degrees(Vec3::from_array(rotation))
    } else {
        transform.rotation_xyzw
    };
    TransformEdit {
        changed: moved || rotated,
        transform: Transform::new(Vec3::from_array(position), quaternion),
        euler_degrees: Vec3::from_array(rotation),
    }
}

/// Keep locked axes unchanged when applying a gizmo edit (upstream
/// `apply_transform_locks`). Returns the merged transform and the rotation
/// angles to keep showing.
pub fn apply_transform_locks(
    current: &Transform,
    edited: &Transform,
    euler_degrees: Option<Vec3>,
    translation_locks: [bool; 3],
    rotation_locks: [bool; 3],
) -> (Transform, Vec3) {
    let mut position = edited.translation;
    for axis in 0..3 {
        if translation_locks[axis] {
            position[axis] = current.translation[axis];
        }
    }
    let current_euler = stable_rotation_degrees(current.rotation_xyzw, euler_degrees);
    let edited_euler = stable_rotation_degrees(edited.rotation_xyzw, Some(current_euler));
    let mut rotation = edited_euler;
    for axis in 0..3 {
        if rotation_locks[axis] {
            rotation[axis] = current_euler[axis];
        }
    }
    let quaternion = if rotation_locks.iter().all(|locked| *locked) {
        current.rotation_xyzw
    } else if rotation_locks.iter().any(|locked| *locked) {
        Transform::quat_from_euler_degrees(rotation)
    } else {
        edited.rotation_xyzw
    };
    (Transform::new(position, quaternion), rotation)
}

/// Draw X, Y, and Z fields with a reset menu and axis locks (upstream
/// `draw_transform_row`).
fn draw_transform_row(ui: &Ui, label: &str, values: &mut [f32; 3], dpi_scale: f32, locks: &mut [bool; 3]) -> bool {
    let _group = ui.begin_group();
    let mut changed = draw_transform_reset(ui, label, values, locks);
    ui.same_line_with_pos(110.0 * dpi_scale);
    for index in 0..3 {
        if index > 0 {
            ui.same_line();
        }
        changed |= draw_transform_axis(ui, label, AXIS_NAMES[index], index, values, locks, dpi_scale);
    }
    changed
}

/// The label with its right-click Reset menu (upstream `_draw_transform_reset`).
fn draw_transform_reset(ui: &Ui, label: &str, values: &mut [f32; 3], locks: &[bool; 3]) -> bool {
    ui.align_text_to_frame_padding();
    let label_text = format!("{label}:");
    let width = ui.calc_text_size(&label_text)[0];
    ui.selectable_config(format!("{label_text}##{label}ResetLabel")).size([width, 0.0]).build();
    let mut changed = false;
    if let Some(_popup) = ui.begin_popup_context_with_label(format!("##{label}Reset")) {
        if ui.menu_item("Reset") {
            for axis in 0..3 {
                if !locks[axis] {
                    changed |= values[axis] != 0.0;
                    values[axis] = 0.0;
                }
            }
        }
    }
    changed
}

/// One axis label (with its lock menu) and float field (upstream `_draw_transform_axis`).
fn draw_transform_axis(
    ui: &Ui,
    label: &str,
    axis_name: &str,
    index: usize,
    values: &mut [f32; 3],
    locks: &mut [bool; 3],
    dpi_scale: f32,
) -> bool {
    let _group = ui.begin_group();
    ui.align_text_to_frame_padding();
    {
        let _color = locks[index].then(|| ui.push_style_color(StyleColor::Text, LOCKED_AXIS_COLOR));
        let width = ui.calc_text_size(axis_name)[0];
        ui.selectable_config(format!("{axis_name}##{label}{axis_name}LockLabel")).size([width, 0.0]).build();
    }
    if let Some(_popup) = ui.begin_popup_context_with_label(format!("##{label}{axis_name}Lock")) {
        let action = if locks[index] { "Unlock" } else { "Lock" };
        if ui.menu_item(format!("{action} {axis_name}")) {
            locks[index] = !locks[index];
        }
    }
    ui.same_line();
    let changed = {
        let _disabled = ui.begin_disabled(locks[index]);
        ui.set_next_item_width(60.0 * dpi_scale);
        ui.input_float(format!("##{label}{axis_name}"), &mut values[index]).build()
    };
    changed
}

/// The angle within ±180 degrees of `reference` that equals `angle` modulo 360
/// (upstream `_nearest_angle`).
fn nearest_angle(angle: f32, reference: f32) -> f32 {
    let delta = (angle - reference + HALF_ROTATION).rem_euclid(FULL_ROTATION);
    reference + delta - HALF_ROTATION
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn quat_degrees(x: f32, y: f32, z: f32) -> Quat {
        Transform::quat_from_euler_degrees(Vec3::new(x, y, z))
    }

    #[test]
    fn clean_display_value_snaps_tiny_angles() {
        assert_eq!(clean_display_value(0.0), 0.0);
        assert_eq!(clean_display_value(4.9e-4), 0.0);
        assert_eq!(clean_display_value(-4.9e-4), 0.0);
        assert_eq!(clean_display_value(0.1), 0.1);
    }

    #[test]
    fn nearest_angle_wraps_to_reference() {
        // 359 is one degree away from -1, not 358.
        assert!((nearest_angle(359.0, -1.0) - -1.0).abs() < 1e-5);
        assert!((nearest_angle(-179.0, 181.0) - 181.0).abs() < 1e-5);
        assert!((nearest_angle(45.0, 45.0) - 45.0).abs() < 1e-5);
    }

    #[test]
    fn stable_rotation_matches_quat_to_rpy_without_reference() {
        let q = quat_degrees(30.0, -45.0, 120.0);
        let euler = stable_rotation_degrees(q, None);
        // Round-trip: the shown angles rebuild the same rotation.
        let rebuilt = Transform::quat_from_euler_degrees(euler);
        assert!(q.abs_diff_eq(rebuilt, 1e-3) || q.abs_diff_eq(-rebuilt, 1e-3), "euler {euler:?}");
    }

    #[test]
    fn stable_rotation_prefers_angles_near_the_reference() {
        // A yaw of 359 is best shown as -1 when the reference shows 0.
        let q = Transform::quat_from_euler_degrees(Vec3::new(0.0, 0.0, -1.0));
        let euler = stable_rotation_degrees(q, Some(Vec3::ZERO));
        assert!(euler.z.abs() <= 1.0 + 1e-4, "expected a small angle, got {euler:?}");
        // With a reference near 359, the same rotation shows as 359 instead.
        let euler = stable_rotation_degrees(q, Some(Vec3::new(0.0, 0.0, 359.0)));
        assert!((euler.z - 359.0).abs() < 1.0, "expected ~359, got {euler:?}");
    }

    #[test]
    fn stable_rotation_picks_the_alternate_representation_when_closer() {
        // Pitch near 90 degrees: roll/yaw of (170, 80, -160) equals (-10, 100, 20).
        let q = quat_degrees(170.0, 80.0, -160.0);
        let euler = stable_rotation_degrees(q, Some(Vec3::new(-10.0, 100.0, 20.0)));
        let distance = (euler - Vec3::new(-10.0, 100.0, 20.0)).length();
        assert!(distance < 1.0, "expected the alternate representation, got {euler:?}");
    }

    #[test]
    fn transforms_equal_treats_negated_quaternion_as_equal() {
        let a = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_z(FRAC_PI_2));
        let b = Transform::new(Vec3::new(1.0, 2.0, 3.0), -a.rotation_xyzw);
        assert!(transforms_equal(&a, &b));
        let moved = Transform::new(Vec3::new(1.1, 2.0, 3.0), a.rotation_xyzw);
        assert!(!transforms_equal(&a, &moved));
    }

    #[test]
    fn apply_transform_locks_keeps_locked_translation_axes() {
        let current = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        let edited = Transform::new(Vec3::new(10.0, 20.0, 30.0), Quat::IDENTITY);
        let (merged, _) = apply_transform_locks(&current, &edited, None, [false, true, false], [false; 3]);
        assert!(merged.translation.abs_diff_eq(Vec3::new(10.0, 2.0, 30.0), 1e-6));
    }

    #[test]
    fn apply_transform_locks_rotation_cases() {
        let current = Transform::new(Vec3::ZERO, Quat::from_rotation_z(0.5));
        let edited = Transform::new(Vec3::ZERO, Quat::from_rotation_x(0.7));
        // No locks: the edited rotation wins.
        let (merged, _) = apply_transform_locks(&current, &edited, None, [false; 3], [false; 3]);
        assert!(merged.rotation_xyzw.abs_diff_eq(edited.rotation_xyzw, 1e-6) || merged.rotation_xyzw.abs_diff_eq(-edited.rotation_xyzw, 1e-6));
        // All locked: the current rotation wins.
        let (merged, _) = apply_transform_locks(&current, &edited, None, [false; 3], [true; 3]);
        assert!(merged.rotation_xyzw.abs_diff_eq(current.rotation_xyzw, 1e-6) || merged.rotation_xyzw.abs_diff_eq(-current.rotation_xyzw, 1e-6));
        // One axis locked (z, the current rotation's only axis): the merged euler keeps
        // the current z and takes the edited x/y, then rebuilds a quaternion.
        let (merged, euler) = apply_transform_locks(&current, &edited, None, [false; 3], [false, false, true]);
        let current_euler = current.euler_degrees();
        assert!((euler.z - current_euler.z).abs() < 0.5, "locked z: {euler:?} vs {current_euler:?}");
        assert!((merged.translation).abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn token_range_constants_match_the_model_contract() {
        // Guards the widget's bounds against a drift of the model contract.
        assert_eq!(MODEL_MIN_TOKEN_COUNT, 6);
        assert_eq!(MODEL_MAX_TOKEN_COUNT, 16);
        assert_eq!(MODEL_POSES_PER_TOKEN, 4);
        // The runtime contract agrees (30 Hz clips, 4 poses per token).
        assert_eq!(motionbricks_rs::runtime::MODEL_POSES_PER_TOKEN as u32, MODEL_POSES_PER_TOKEN);
    }

    #[test]
    fn sample_frame_at_rounds_and_clamps() {
        assert_eq!(sample_frame_at(0.0, 30.0, 99), 0);
        assert_eq!(sample_frame_at(1.0, 30.0, 99), 30);
        assert_eq!(sample_frame_at(31.0 / 30.0, 30.0, 99), 31);
        // Nearest frame wins over truncation.
        assert_eq!(sample_frame_at(31.4 / 30.0, 30.0, 99), 31);
        assert_eq!(sample_frame_at(31.6 / 30.0, 30.0, 99), 32);
        // Never past the clip, never negative.
        assert_eq!(sample_frame_at(100.0, 30.0, 99), 99);
        assert_eq!(sample_frame_at(-1.0, 30.0, 99), 0);
    }

    #[test]
    fn clamp_token_range_enforces_the_model_contract() {
        assert_eq!(clamp_token_range(6, 14), (6, 14));
        // Bounds are the model's.
        assert_eq!(clamp_token_range(1, 100), (MODEL_MIN_TOKEN_COUNT, MODEL_MAX_TOKEN_COUNT));
        // The maximum never drops below the minimum.
        assert_eq!(clamp_token_range(10, 8), (10, 10));
        assert_eq!(clamp_token_range(6, 6), (6, 6));
    }
}
