import type { InvocationSourceId } from '@workbench/invocationContracts';
import type { BuiltInLayoutPresetId } from '@workbench/layoutContracts';

/**
 * Apply URL intent once to the fresh draft: each intent chooses a built-in arrangement and the invocation source.
 * Keep built-in ids here so the route validates without loading full preset snapshots.
 */

export type LaunchpadIntentId = 'generate' | 'canvas' | 'upscale' | 'video' | 'workflow';

export interface LaunchpadIntent {
  id: LaunchpadIntentId;
  presetId: BuiltInLayoutPresetId;
  sourceId: InvocationSourceId;
}

export const LAUNCHPAD_INTENT_IDS: readonly LaunchpadIntentId[] = [
  'generate',
  'canvas',
  'upscale',
  'video',
  'workflow',
];

const INTENTS: Record<LaunchpadIntentId, LaunchpadIntent> = {
  canvas: { id: 'canvas', presetId: 'edit', sourceId: 'canvas' },
  generate: { id: 'generate', presetId: 'compose', sourceId: 'generate' },
  upscale: { id: 'upscale', presetId: 'compose', sourceId: 'upscale' },
  video: { id: 'video', presetId: 'video', sourceId: 'video' },
  workflow: { id: 'workflow', presetId: 'automate', sourceId: 'workflow' },
};

export const isLaunchpadIntentId = (value: unknown): value is LaunchpadIntentId =>
  typeof value === 'string' && LAUNCHPAD_INTENT_IDS.includes(value as LaunchpadIntentId);

/**
 * Share built-in labels without loading widget-region snapshots; layoutPresets derives its labels from these maps.
 * The keys name what a preset is called in the UI; the English labels are the shipped defaults that a stored rename
 * is compared with. Ids stay as they were (`automate`): layouts and saved bindings persist them.
 */
export const BUILT_IN_LAYOUT_PRESET_LABEL_KEYS: Record<BuiltInLayoutPresetId, string> = {
  automate: 'topbar.presets.builtInLabels.automate',
  compose: 'topbar.presets.builtInLabels.compose',
  edit: 'topbar.presets.builtInLabels.edit',
  video: 'topbar.presets.builtInLabels.video',
};

export const BUILT_IN_LAYOUT_PRESET_LABELS: Record<BuiltInLayoutPresetId, string> = {
  automate: 'Workflow',
  compose: 'Compose',
  edit: 'Edit',
  video: 'Video',
};

/**
 * `null` for anything unrecognised — a hand-edited or stale URL should open a
 * plain draft rather than fail.
 */
export const resolveLaunchpadIntent = (value: unknown): LaunchpadIntent | null =>
  isLaunchpadIntentId(value) ? INTENTS[value] : null;
