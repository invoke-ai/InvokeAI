import { describe, expect, it } from 'vitest';

import { builtInLayoutPresetDescriptors, layoutPresets } from './layoutPresets';

describe('built-in layout preset descriptors', () => {
  it('owns the preset, presentation, and command metadata in strip order', () => {
    expect(
      builtInLayoutPresetDescriptors.map(({ defaultKeys, hotkeyId, iconId, preset, tooltip }) => ({
        defaultKeys,
        hotkeyId,
        iconId,
        label: preset.label,
        presetId: preset.id,
        tooltip,
      }))
    ).toEqual([
      {
        defaultKeys: ['alt+1'],
        hotkeyId: 'selectComposePreset',
        iconId: 'type',
        label: 'Compose',
        presetId: 'compose',
        tooltip: 'Text to image',
      },
      {
        defaultKeys: ['alt+2'],
        hotkeyId: 'selectEditPreset',
        iconId: 'layers',
        label: 'Edit',
        presetId: 'edit',
        tooltip: 'Canvas editing',
      },
      {
        defaultKeys: ['alt+3'],
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
