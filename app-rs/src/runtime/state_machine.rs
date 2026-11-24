// SPDX-License-Identifier: Apache-2.0
// Rust port of NVlabs/motionbricks app/ (the MotionBricks applications).
// Original model: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
// Modifications Copyright 2026 The motionbricks-mlx Authors (see NOTICE).
//! The state machine that switches a character between idle, locomotion, and
//! interactions, ported from upstream `app/runtime/common/state_machine.py`.
//!
//! A state change exits the old state before it enters the new one. Because the
//! machine owns its states, it moves the active state out while it runs
//! (`take`/`put`); a state asks for a transition through a `deferred` target the
//! machine applies once the state is back in place.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use super::behaviors::SmartPrimitiveBehavior;
use super::character::{Character, ElemOf, EngineAdapter};
use super::motion::{
    AnimationClip, CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE, LocomotionCommand, LocomotionModeRuntime, LocomotionSettings,
    PoseElem,
};
use super::states::{IdleState, InteractionState, LocomotionState, State};

/// A crouching character is shorter, so its probe box is this fraction as tall.
const CROUCH_PROBE_HEIGHT_SCALE: f32 = 0.6;

/// Which movement pair is active: standing or crouching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MovementPair {
    Standing,
    Crouching,
}

/// Which of the machine's states is current.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveState {
    StandingIdle,
    StandingLocomotion,
    CrouchingIdle,
    CrouchingLocomotion,
    Interaction,
}

/// Switch between idle, locomotion, and interactions while standing or crouching
/// (upstream `StateMachine`).
///
/// The caller loads the locomotion modes and the idle clips. The states generate
/// motion, and this machine decides which one runs.
pub struct StateMachine<A: EngineAdapter> {
    standing_idle: State<A>,
    standing_locomotion: State<A>,
    crouching_idle: State<A>,
    crouching_locomotion: State<A>,
    interaction: State<A>,
    active: ActiveState,
    active_pair: MovementPair,
    pair_by_mode: HashMap<String, MovementPair>,
    style_modes: Vec<String>,
}

impl<A: EngineAdapter> StateMachine<A>
where
    ElemOf<A>: PoseElem,
{
    /// Create the idle and locomotion states for standing and crouching.
    ///
    /// The interaction state is shared by both. The state machine starts in the idle
    /// state of `initial_mode`. Call [`StateMachine::start`] to begin its idle loop.
    ///
    /// `mode_runtimes` must include Forward, Strafing, and Crouching. Any other mode is
    /// a standing style mode, listed in `style_modes` in the order given. `pose_len`
    /// sizes the target pose buffers.
    pub fn new(
        mode_runtimes: Vec<LocomotionModeRuntime<ElemOf<A>>>,
        initial_mode: &str,
        standing_idle_clips: Vec<AnimationClip<A::Buffer>>,
        crouching_idle_clips: Vec<AnimationClip<A::Buffer>>,
        pose_len: usize,
    ) -> Result<Self> {
        let mut seen = HashSet::new();
        for runtime in &mode_runtimes {
            if !seen.insert(runtime.definition.name.as_str()) {
                bail!("Locomotion mode names must be unique.");
            }
        }
        // Any mode beyond the core three is a standing style mode, in the order given.
        let given_names: Vec<String> = mode_runtimes.iter().map(|runtime| runtime.definition.name.clone()).collect();
        let modes_by_name: HashMap<String, LocomotionModeRuntime<ElemOf<A>>> = mode_runtimes.into_iter().map(|runtime| (runtime.definition.name.clone(), runtime)).collect();
        const CORE_NAMES: [&str; 3] = [LINEAR_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, CROUCH_LOCOMOTION_MODE];
        let missing: Vec<&str> = CORE_NAMES.into_iter().filter(|name| !modes_by_name.contains_key(*name)).collect();
        if !missing.is_empty() {
            bail!("Required locomotion modes are missing: {}.", missing.join(", "));
        }
        if !modes_by_name.contains_key(initial_mode) {
            bail!("Unknown initial locomotion mode {initial_mode:?}.");
        }

        let style_modes: Vec<String> = given_names.into_iter().filter(|name| !CORE_NAMES.contains(&name.as_str())).collect();
        let standing_names: Vec<&str> = [LINEAR_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE].into_iter().chain(style_modes.iter().map(String::as_str)).collect();
        let pick = |modes_by_name: &mut HashMap<String, LocomotionModeRuntime<ElemOf<A>>>, names: &[&str]| names.iter().map(|name| modes_by_name.remove(*name).expect("checked above")).collect::<Vec<_>>();
        let mut modes_by_name = modes_by_name;
        let standing_runtimes = pick(&mut modes_by_name, &standing_names);
        let crouch_runtime = modes_by_name.remove(CROUCH_LOCOMOTION_MODE).expect("checked above");

        let standing_locomotion = LocomotionState::new(
            "Locomotion",
            standing_runtimes,
            if initial_mode != CROUCH_LOCOMOTION_MODE { initial_mode } else { LINEAR_LOCOMOTION_MODE },
            pose_len,
            LocomotionSettings::default(),
        )?;
        let crouching_locomotion = LocomotionState::new("Crouching Locomotion", vec![crouch_runtime], CROUCH_LOCOMOTION_MODE, pose_len, LocomotionSettings::default())?;

        let mut pair_by_mode = HashMap::new();
        for name in standing_names {
            pair_by_mode.insert(name.to_string(), MovementPair::Standing);
        }
        pair_by_mode.insert(CROUCH_LOCOMOTION_MODE.to_string(), MovementPair::Crouching);
        let active_pair = pair_by_mode[initial_mode];

        Ok(StateMachine {
            standing_idle: State::Idle(IdleState::new("Idle", standing_idle_clips)?),
            standing_locomotion: State::Locomotion(standing_locomotion),
            crouching_idle: State::Idle(IdleState::new("Crouching Idle", crouching_idle_clips)?),
            crouching_locomotion: State::Locomotion(crouching_locomotion),
            interaction: State::Interaction(InteractionState::new()),
            active: match active_pair {
                MovementPair::Standing => ActiveState::StandingIdle,
                MovementPair::Crouching => ActiveState::CrouchingIdle,
            },
            active_pair,
            pair_by_mode,
            style_modes,
        })
    }

    /// The placeholder a character leaves behind while the machine is moved out.
    pub(crate) fn placeholder() -> Self {
        StateMachine {
            standing_idle: State::placeholder(),
            standing_locomotion: State::placeholder(),
            crouching_idle: State::placeholder(),
            crouching_locomotion: State::placeholder(),
            interaction: State::placeholder(),
            active: ActiveState::StandingIdle,
            active_pair: MovementPair::Standing,
            pair_by_mode: HashMap::new(),
            style_modes: Vec::new(),
        }
    }

    fn state_ref(&self, key: ActiveState) -> &State<A> {
        match key {
            ActiveState::StandingIdle => &self.standing_idle,
            ActiveState::StandingLocomotion => &self.standing_locomotion,
            ActiveState::CrouchingIdle => &self.crouching_idle,
            ActiveState::CrouchingLocomotion => &self.crouching_locomotion,
            ActiveState::Interaction => &self.interaction,
        }
    }

    fn state_mut(&mut self, key: ActiveState) -> &mut State<A> {
        match key {
            ActiveState::StandingIdle => &mut self.standing_idle,
            ActiveState::StandingLocomotion => &mut self.standing_locomotion,
            ActiveState::CrouchingIdle => &mut self.crouching_idle,
            ActiveState::CrouchingLocomotion => &mut self.crouching_locomotion,
            ActiveState::Interaction => &mut self.interaction,
        }
    }

    fn take(&mut self, key: ActiveState) -> State<A> {
        std::mem::replace(self.state_mut(key), State::placeholder())
    }

    fn put(&mut self, key: ActiveState, state: State<A>) {
        *self.state_mut(key) = state;
    }

    /// The idle state key of the active movement pair.
    pub(crate) fn idle_key(&self) -> ActiveState {
        match self.active_pair {
            MovementPair::Standing => ActiveState::StandingIdle,
            MovementPair::Crouching => ActiveState::CrouchingIdle,
        }
    }

    /// The locomotion state key of the active movement pair.
    pub(crate) fn locomotion_key(&self) -> ActiveState {
        match self.active_pair {
            MovementPair::Standing => ActiveState::StandingLocomotion,
            MovementPair::Crouching => ActiveState::CrouchingLocomotion,
        }
    }

    /// Locomotion when there is movement input, or idle otherwise.
    pub(crate) fn resume_key(&self, character: &Character<A>) -> ActiveState {
        if character.movement.input_speed > 0.0 { self.locomotion_key() } else { self.idle_key() }
    }

    fn locomotion_state(&self, pair: MovementPair) -> &LocomotionState<ElemOf<A>> {
        let key = match pair {
            MovementPair::Standing => ActiveState::StandingLocomotion,
            MovementPair::Crouching => ActiveState::CrouchingLocomotion,
        };
        match self.state_ref(key) {
            State::Locomotion(state) => state,
            _ => unreachable!("locomotion keys hold locomotion states"),
        }
    }

    fn locomotion_state_mut(&mut self, pair: MovementPair) -> &mut LocomotionState<ElemOf<A>> {
        let key = match pair {
            MovementPair::Standing => ActiveState::StandingLocomotion,
            MovementPair::Crouching => ActiveState::CrouchingLocomotion,
        };
        match self.state_mut(key) {
            State::Locomotion(state) => state,
            _ => unreachable!("locomotion keys hold locomotion states"),
        }
    }

    /// Current state.
    pub fn state(&self) -> ActiveState {
        self.active
    }

    /// Name of the current state ("Idle", "Locomotion", "Crouching Idle",
    /// "Crouching Locomotion", or "Interaction").
    pub fn state_name(&self) -> &'static str {
        self.state_ref(self.active).name()
    }

    /// Locomotion state of the active movement pair, standing or crouching.
    pub fn active_locomotion_state(&self) -> &LocomotionState<ElemOf<A>> {
        self.locomotion_state(self.active_pair)
    }

    /// Scale of the probe box height for the active movement pair.
    pub fn probe_box_height_scale(&self) -> f32 {
        match self.active_pair {
            MovementPair::Standing => 1.0,
            MovementPair::Crouching => CROUCH_PROBE_HEIGHT_SCALE,
        }
    }

    /// Whether a locomotion state is current.
    pub fn is_locomoting(&self) -> bool {
        matches!(self.active, ActiveState::StandingLocomotion | ActiveState::CrouchingLocomotion)
    }

    /// Whether the interaction state is current.
    pub fn is_interacting(&self) -> bool {
        self.active == ActiveState::Interaction
    }

    /// The standing style modes, in the order the modes were given.
    pub fn style_modes(&self) -> &[String] {
        &self.style_modes
    }

    /// The shared interaction state.
    pub fn interaction_state(&self) -> &InteractionState<A> {
        match &self.interaction {
            State::Interaction(state) => state,
            _ => unreachable!("the interaction slot holds the interaction state"),
        }
    }

    /// Start the character in the initial idle loop.
    pub fn start(&mut self, character: &mut Character<A>) -> Result<()> {
        let key = self.idle_key();
        let mut state = self.take(key);
        let result = state.start(character);
        self.put(key, state);
        result
    }

    /// Exit the current state and enter `next_state`, unless it is already current.
    pub fn transition_to(&mut self, character: &mut Character<A>, next_state: ActiveState) -> Result<()> {
        if next_state == self.active {
            return Ok(());
        }
        let previous = self.active;
        let mut state = self.take(previous);
        state.exit(character);
        self.put(previous, state);

        self.active = next_state;
        let mut deferred = None;
        let mut state = self.take(next_state);
        let result = state.enter(character, self, &mut deferred);
        self.put(next_state, state);
        result?;
        if let Some(next) = deferred {
            self.transition_to(character, next)?;
        }
        Ok(())
    }

    /// Switch modes as the command asks, smooth the movement, then update the current
    /// state.
    ///
    /// Movement smooths in every state, so it carries into the next one.
    pub fn update(&mut self, character: &mut Character<A>, delta_time: f32, command: &LocomotionCommand) -> Result<()> {
        let previous_pair = self.active_pair;
        let previous_mode = self.active_locomotion_state().active_mode().clone();
        if let Some(requested) = &command.requested_mode {
            let Some(&pair) = self.pair_by_mode.get(requested) else {
                bail!("Unknown locomotion mode {requested:?}.");
            };
            self.active_pair = pair;
            self.locomotion_state_mut(pair).select_mode(requested)?;
        }

        let mode = self.active_locomotion_state().active_mode().clone();
        character.movement.update(command, &mode, delta_time, if mode.name != previous_mode.name { Some(&previous_mode) } else { None });

        let pair_changed = previous_pair != self.active_pair;
        let previous_idle = match previous_pair {
            MovementPair::Standing => ActiveState::StandingIdle,
            MovementPair::Crouching => ActiveState::CrouchingIdle,
        };
        let previous_locomotion = match previous_pair {
            MovementPair::Standing => ActiveState::StandingLocomotion,
            MovementPair::Crouching => ActiveState::CrouchingLocomotion,
        };
        if pair_changed && (self.active == previous_idle || self.active == previous_locomotion) {
            self.resume_movement(character)?;
        }

        let key = self.active;
        let mut deferred = None;
        let mut state = self.take(key);
        let result = state.update(character, self, delta_time, &mut deferred);
        self.put(key, state);
        result?;
        if let Some(next) = deferred {
            self.transition_to(character, next)?;
        }
        Ok(())
    }

    /// Switch to the idle state of the active movement pair.
    pub fn transition_to_idle(&mut self, character: &mut Character<A>) -> Result<()> {
        let key = self.idle_key();
        self.transition_to(character, key)
    }

    /// Switch to the locomotion state of the active movement pair.
    pub fn transition_to_locomotion(&mut self, character: &mut Character<A>) -> Result<()> {
        let key = self.locomotion_key();
        self.transition_to(character, key)
    }

    /// Switch to locomotion when there is movement input, or to idle otherwise.
    pub fn resume_movement(&mut self, character: &mut Character<A>) -> Result<()> {
        let key = self.resume_key(character);
        self.transition_to(character, key)
    }

    /// Let a Smart Primitive control the character until it calls finish.
    pub fn start_interaction(&mut self, character: &mut Character<A>, primitive: Box<dyn SmartPrimitiveBehavior<ElemOf<A>>>) -> Result<()> {
        if self.is_interacting() {
            bail!("Cannot start an interaction while another is active.");
        }
        match &mut self.interaction {
            State::Interaction(state) => state.configure(character, primitive)?,
            _ => unreachable!("the interaction slot holds the interaction state"),
        }
        match self.transition_to(character, ActiveState::Interaction) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.resume_movement(character)?;
                Err(error)
            }
        }
    }

    /// Pass an interaction press to the active Smart Primitive behavior.
    pub fn interact(&mut self, character: &mut Character<A>) -> Result<()> {
        if !self.is_interacting() {
            return Ok(());
        }
        let mut deferred = None;
        let mut state = self.take(ActiveState::Interaction);
        let result = state.interact(character, self, &mut deferred);
        self.put(ActiveState::Interaction, state);
        result?;
        if let Some(next) = deferred {
            self.transition_to(character, next)?;
        }
        Ok(())
    }
}
