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
 * `null` for anything unrecognised — a hand-edited or stale URL should open a
 * plain draft rather than fail.
 */
export const resolveLaunchpadIntent = (value: unknown): LaunchpadIntent | null =>
  isLaunchpadIntentId(value) ? INTENTS[value] : null;
