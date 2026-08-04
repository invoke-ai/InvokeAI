import { isLaunchpadIntentId, LAUNCHPAD_INTENT_IDS, resolveLaunchpadIntent } from '@workbench/launchpad/intents';
import { builtInLayoutPresetDescriptors } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

describe('launchpad intents', () => {
  it('resolves every advertised id', () => {
    for (const id of LAUNCHPAD_INTENT_IDS) {
      expect(resolveLaunchpadIntent(id)?.id).toBe(id);
    }
  });

  it('names a real built-in preset for every intent', () => {
    const presetIds = new Set(builtInLayoutPresetDescriptors.map((descriptor) => descriptor.preset.id));

    for (const id of LAUNCHPAD_INTENT_IDS) {
      expect(presetIds).toContain(resolveLaunchpadIntent(id)?.presetId);
    }
  });

  it('opens text-to-image in Compose and node work in Automate', () => {
    expect(resolveLaunchpadIntent('generate')).toEqual({ id: 'generate', presetId: 'compose', sourceId: 'generate' });
    expect(resolveLaunchpadIntent('workflow')).toEqual({ id: 'workflow', presetId: 'automate', sourceId: 'workflow' });
  });

  it('sets a source that a preset alone could not imply', () => {
    // Compose hosts both generate and upscale, so the intent has to say which.
    const generate = resolveLaunchpadIntent('generate');
    const upscale = resolveLaunchpadIntent('upscale');

    expect(generate?.presetId).toBe(upscale?.presetId);
    expect(generate?.sourceId).not.toBe(upscale?.sourceId);
  });

  it('falls back to no intent for a stale or hand-edited value', () => {
    expect(resolveLaunchpadIntent('sideways')).toBeNull();
    expect(resolveLaunchpadIntent(undefined)).toBeNull();
    expect(resolveLaunchpadIntent(42)).toBeNull();
    expect(isLaunchpadIntentId('generate')).toBe(true);
    expect(isLaunchpadIntentId('')).toBe(false);
  });
});
