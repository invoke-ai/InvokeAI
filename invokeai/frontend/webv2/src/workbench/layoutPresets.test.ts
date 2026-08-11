import { describe, expect, it } from 'vitest';

import { builtInLayoutPresetDescriptors, layoutPresets } from './layoutPresets';

describe('built-in layout preset descriptors', () => {
  it('owns the preset, presentation, and command metadata in strip order', () => {
    expect(
      builtInLayoutPresetDescriptors.map(({ defaultKeys, hotkeyId, preset, tooltip }) => ({
        defaultKeys,
        defaultRoute: preset.defaultRoute,
        hotkeyId,
        iconId: preset.iconId,
        label: preset.label,
        presetId: preset.id,
        tooltip,
      }))
    ).toEqual([
      {
        defaultKeys: ['alt+1'],
        defaultRoute: { destination: 'gallery', sourceId: 'generate' },
        hotkeyId: 'selectComposePreset',
        iconId: 'type',
        label: 'Compose',
        presetId: 'compose',
        tooltip: 'Text to image',
      },
      {
        defaultKeys: ['alt+2'],
        defaultRoute: { destination: 'canvas', sourceId: 'canvas' },
        hotkeyId: 'selectEditPreset',
        iconId: 'layers',
        label: 'Edit',
        presetId: 'edit',
        tooltip: 'Canvas editing',
      },
      {
        defaultKeys: ['alt+3'],
        defaultRoute: { destination: 'gallery', sourceId: 'workflow' },
        hotkeyId: 'selectAutomatePreset',
        iconId: 'workflow',
        label: 'Automate',
        presetId: 'automate',
        tooltip: 'Node workflows',
      },
    ]);
    expect(layoutPresets).toEqual(builtInLayoutPresetDescriptors.map(({ preset }) => preset));
  });
});
