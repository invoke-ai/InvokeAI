import { describe, expect, it } from 'vitest';

import { builtInLayoutPresetDescriptors, layoutPresets } from './layoutPresets';

describe('built-in layout preset descriptors', () => {
  it('owns the preset, presentation, and command metadata in strip order', () => {
    expect(
      builtInLayoutPresetDescriptors.map(({ defaultKeys, hotkeyId, preset }) => ({
        defaultKeys,
        defaultRoute: preset.defaultRoute,
        hotkeyId,
        iconId: preset.iconId,
        label: preset.label,
        presetId: preset.id,
      }))
    ).toEqual([
      {
        defaultKeys: ['alt+1'],
        defaultRoute: { destination: 'gallery', sourceId: 'generate' },
        hotkeyId: 'selectComposePreset',
        iconId: 'type',
        label: 'Compose',
        presetId: 'compose',
      },
      {
        defaultKeys: ['alt+2'],
        defaultRoute: { destination: 'canvas', sourceId: 'canvas' },
        hotkeyId: 'selectEditPreset',
        iconId: 'layers',
        label: 'Edit',
        presetId: 'edit',
      },
      {
        defaultKeys: ['alt+3'],
        defaultRoute: { destination: 'gallery', sourceId: 'workflow' },
        hotkeyId: 'selectAutomatePreset',
        iconId: 'workflow',
        label: 'Automate',
        presetId: 'automate',
      },
    ]);
    expect(layoutPresets).toEqual(builtInLayoutPresetDescriptors.map(({ preset }) => preset));
  });

  it('ships task-focused widget sets in interaction-priority order', () => {
    const summaries = Object.fromEntries(
      layoutPresets.map(({ id, snapshot }) => [
        id,
        {
          active: {
            bottom: snapshot.widgetRegions.bottom.activeInstanceId,
            center: snapshot.widgetRegions.center.activeInstanceId,
            left: snapshot.widgetRegions.left.activeInstanceId,
            right: snapshot.widgetRegions.right.activeInstanceId,
          },
          bottom: snapshot.widgetRegions.bottom.instanceIds,
          center: snapshot.widgetRegions.center.instanceIds,
          left: snapshot.widgetRegions.left.instanceIds,
          panels: snapshot.layout.panels,
          right: snapshot.widgetRegions.right.instanceIds,
        },
      ])
    );

    expect(summaries).toEqual({
      automate: {
        active: { bottom: 'gallery:bottom', center: 'workflow:center', left: 'workflow', right: 'queue' },
        bottom: ['server-status', 'queue-status', 'gallery:bottom', 'notifications', 'autosave-status'],
        center: ['workflow:center', 'preview'],
        left: ['workflow'],
        panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
        right: ['queue', 'preview', 'gallery', 'image-map'],
      },
      compose: {
        active: { bottom: 'gallery:bottom', center: 'preview', left: 'generate', right: 'gallery' },
        bottom: ['server-status', 'queue-status', 'gallery:bottom', 'notifications', 'autosave-status'],
        // Compose keeps Gallery as a center view: the retired `gallery` preset
        // migrates here (see legacyLayoutPresetIds) precisely because it was
        // "Compose with the center view swapped", so the swap has to remain
        // available.
        center: ['preview', 'gallery:center'],
        left: ['generate', 'upscale'],
        panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
        right: ['gallery', 'image-map', 'queue'],
      },
      edit: {
        active: { bottom: 'gallery:bottom', center: 'canvas', left: 'generate', right: 'layers' },
        bottom: ['server-status', 'queue-status', 'gallery:bottom', 'notifications', 'autosave-status'],
        center: ['canvas', 'preview'],
        left: ['generate', 'upscale'],
        panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
        right: ['layers', 'preview', 'gallery', 'image-map', 'queue'],
      },
    });
  });
});
