import type { InvocationSourceId } from '@workbench/invocationContracts';
import type { BuiltInLayoutPresetId } from '@workbench/layoutContracts';

/**
 * What the user said they wanted to make, translated into how the editor
 * should open.
 *
 * A layout preset supplies the account's default route, while an intent names
 * its source explicitly because it can be more specific. For example, Generate
 * and Upscale share Compose but need different invocation sources.
 *
 * This is the whole contract behind `/app?new=true&intent=…`: the Launchpad
 * writes the id into the URL, and the editor's session controller applies it
 * once, on the fresh draft.
 *
 * `?preset=` is the same contract one level lower: it names an arrangement
 * directly, for when the user picked a layout rather than a kind of work. The
 * id list lives here rather than being derived from `layoutPresets` so that
 * validating the URL costs the editor route nothing — this module is already
 * shared between both routes, the preset table is not.
 */

export type LaunchpadIntentId = 'generate' | 'canvas' | 'upscale' | 'workflow';

export interface LaunchpadIntent {
  id: LaunchpadIntentId;
  presetId: BuiltInLayoutPresetId;
  sourceId: InvocationSourceId;
}

export const LAUNCHPAD_INTENT_IDS: readonly LaunchpadIntentId[] = ['generate', 'canvas', 'upscale', 'workflow'];

const INTENTS: Record<LaunchpadIntentId, LaunchpadIntent> = {
  canvas: { id: 'canvas', presetId: 'edit', sourceId: 'canvas' },
  generate: { id: 'generate', presetId: 'compose', sourceId: 'generate' },
  upscale: { id: 'upscale', presetId: 'compose', sourceId: 'upscale' },
  workflow: { id: 'workflow', presetId: 'automate', sourceId: 'workflow' },
};

export const isLaunchpadIntentId = (value: unknown): value is LaunchpadIntentId =>
  typeof value === 'string' && LAUNCHPAD_INTENT_IDS.includes(value as LaunchpadIntentId);

/**
 * The arrangements the Launchpad can start a draft in. Built-ins only: custom
 * presets live in the account state that only the mounted workbench holds, and
 * the Launchpad renders outside that provider.
 */
export const LAUNCHPAD_LAYOUT_IDS: readonly BuiltInLayoutPresetId[] = ['compose', 'edit', 'automate'];

/**
 * Display names for the built-in arrangements, kept here rather than read off
 * `layoutPresets` so the Launchpad can name them without pulling that table —
 * which carries three full widget-region snapshots — onto its route chunk.
 * `layoutPresets` builds its own labels from this map, so there is still one
 * definition.
 */
export const BUILT_IN_LAYOUT_PRESET_LABELS: Record<BuiltInLayoutPresetId, string> = {
  automate: 'Automate',
  compose: 'Compose',
  edit: 'Edit',
};

export const isLaunchpadLayoutId = (value: unknown): value is BuiltInLayoutPresetId =>
  typeof value === 'string' && LAUNCHPAD_LAYOUT_IDS.includes(value as BuiltInLayoutPresetId);

/**
 * `null` for anything unrecognised — a hand-edited or stale URL should open a
 * plain draft rather than fail.
 */
export const resolveLaunchpadIntent = (value: unknown): LaunchpadIntent | null =>
  isLaunchpadIntentId(value) ? INTENTS[value] : null;
