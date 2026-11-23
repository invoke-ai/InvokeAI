import { describe, expect, it } from 'vitest';

import { getLayoutPresetLabel, layoutPresets } from './layoutPresets';

describe('built-in layout preset descriptors', () => {
  it('has a widget instance for every instance id a preset places', () => {
    for (const preset of layoutPresets) {
      for (const region of Object.values(preset.snapshot.widgetRegions)) {
        for (const instanceId of region.instanceIds) {
          expect(preset.snapshot.widgetInstances[instanceId], `${preset.id} ${instanceId}`).toBeDefined();
        }
      }
    }
  });

  it('names each built-in preset through the locale, Automate as Workflow', () => {
    const t = (key: string) => `t:${key}`;

    expect(layoutPresets.map((preset) => getLayoutPresetLabel(preset, t))).toEqual([
      't:topbar.presets.builtInLabels.compose',
      't:topbar.presets.builtInLabels.edit',
      't:topbar.presets.builtInLabels.video',
      't:topbar.presets.builtInLabels.automate',
    ]);
    expect(getLayoutPresetLabel({ label: 'My layout' }, t)).toBe('My layout');
  });
});
