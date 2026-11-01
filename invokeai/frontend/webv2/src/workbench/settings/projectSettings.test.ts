import { firstPartyWidgetManifests } from '@workbench/widgets/manifests';
import { describe, expect, it } from 'vitest';

import type { ProjectSettings } from './contracts';

import {
  BUILT_IN_PROJECT_SETTINGS,
  getEffectiveProjectSettings,
  getLayeredKeys,
  getLayeredSettings,
  LAYERED_PROJECT_SETTING_KEYS,
  normalizeProjectSettings,
  PROJECT_SETTINGS_OWNER,
} from './projectSettings';

describe('normalizeProjectSettings', () => {
  it('starts a project saved before Auto-switch Off, whatever its old progress toggle said', () => {
    type Legacy = Parameters<typeof normalizeProjectSettings>[0] & { showProgressImagesInViewer?: boolean };
    const legacy = (settings: Legacy) => normalizeProjectSettings(settings);

    expect(legacy({}).viewerAutoSwitch).toBe('off');
    expect(legacy({ showProgressImagesInViewer: true }).viewerAutoSwitch).toBe('off');
    expect(legacy({ showProgressImagesInViewer: false }).viewerAutoSwitch).toBe('off');
    expect(legacy({ showProgressImagesInViewer: true })).not.toHaveProperty('showProgressImagesInViewer');
  });

  it('keeps a saved Auto-switch mode and reads anything else as Off', () => {
    for (const mode of ['off', 'latest', 'progress'] as const) {
      expect(normalizeProjectSettings({ viewerAutoSwitch: mode }).viewerAutoSwitch).toBe(mode);
    }
    expect(normalizeProjectSettings({ viewerAutoSwitch: 'always' as unknown as 'off' }).viewerAutoSwitch).toBe('off');
  });

  it('starts new projects on In progress', () => {
    expect(normalizeProjectSettings(BUILT_IN_PROJECT_SETTINGS).viewerAutoSwitch).toBe('progress');
  });

  it('keeps the generation flags a project pinned and leaves the others to follow the default', () => {
    expect(normalizeProjectSettings({ viewerAutoSwitch: 'off' })).toEqual({ viewerAutoSwitch: 'off' });
    expect(
      normalizeProjectSettings({
        useCpuNoise: 'yes' as unknown as boolean,
        useNsfwChecker: false,
        useWatermark: true,
        viewerAutoSwitch: 'off',
      })
    ).toEqual({ useNsfwChecker: false, useWatermark: true, viewerAutoSwitch: 'off' });
  });

  /**
   * Discard legacy project highlighting/attention settings and the retired per-project antialias flag so imports
   * cannot overwrite personal preferences.
   */
  it('drops preferences that older documents still carry in project settings', () => {
    const normalized = normalizeProjectSettings({
      antialiasProgressImages: true,
      preferNumericAttentionStyle: true,
      showPromptSyntaxHighlighting: true,
    } as Partial<ProjectSettings>);

    expect(normalized).toEqual({ viewerAutoSwitch: 'off' });
  });
});

describe('getEffectiveProjectSettings', () => {
  it('reads each flag from the project pin, then the user default, then the built-in', () => {
    const settings: ProjectSettings = { useNsfwChecker: false, viewerAutoSwitch: 'latest' };
    const defaults = { useCpuNoise: false, useNsfwChecker: true };

    expect(getEffectiveProjectSettings(settings, defaults)).toEqual({
      useCpuNoise: false,
      useNsfwChecker: false,
      useWatermark: false,
      viewerAutoSwitch: 'latest',
    });
    expect(getEffectiveProjectSettings(settings, undefined)).toEqual({
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
      viewerAutoSwitch: 'latest',
    });
  });

  it('ignores a default that is not a boolean', () => {
    expect(getEffectiveProjectSettings({ viewerAutoSwitch: 'off' }, { useWatermark: 'on' }).useWatermark).toBe(false);
  });

  it('returns the same object for the same settings and defaults', () => {
    const settings: ProjectSettings = { viewerAutoSwitch: 'off' };
    const defaults = { useWatermark: true };
    const first = getEffectiveProjectSettings(settings, defaults);

    expect(getEffectiveProjectSettings(settings, defaults)).toBe(first);
    expect(getEffectiveProjectSettings(settings, { useWatermark: true })).not.toBe(first);
  });
});

describe('getLayeredSettings', () => {
  const contributionKeys = (fields: readonly { id: string }[]) =>
    fields.map((field) => (field.id === 'imageSize' ? 'imageDensityPercent' : field.id)).sort();

  it('lists exactly the keys each settings contribution edits', () => {
    const table = getLayeredSettings();
    const manifests = Object.fromEntries(firstPartyWidgetManifests.map((manifest) => [manifest.id, manifest]));
    const layeredFields = (typeId: string) =>
      manifests[typeId]!.settings!.fields.filter((field) => field.scope === 'project');

    for (const typeId of Object.keys(table.widgetValues)) {
      expect(Object.keys(table.widgetValues[typeId]!).sort(), typeId).toEqual(contributionKeys(layeredFields(typeId)));
    }
    expect(Object.keys(table.projectSettings).sort()).toEqual(contributionKeys(layeredFields(PROJECT_SETTINGS_OWNER)));
    expect(Object.keys(table.projectSettings).sort()).toEqual([...LAYERED_PROJECT_SETTING_KEYS].sort());
    // Every manifest with layered fields is in the table.
    const withLayered = firstPartyWidgetManifests
      .filter((manifest) => manifest.settings?.fields.some((field) => field.scope === 'project'))
      .map((manifest) => manifest.id)
      .sort();
    expect(withLayered).toEqual([...Object.keys(table.widgetValues), PROJECT_SETTINGS_OWNER].sort());
  });

  it('reads an absent value as the built-in, as each owner does', () => {
    const table = getLayeredSettings();
    for (const owner of [...Object.values(table.widgetValues), table.projectSettings]) {
      for (const entry of Object.values(owner)) {
        expect(entry.read(undefined)).toBe(entry.builtIn);
      }
    }
  });

  it('maps the Generate owner to the project settings and widget types to their values', () => {
    expect(getLayeredKeys(PROJECT_SETTINGS_OWNER)).toBe(getLayeredSettings().projectSettings);
    expect(getLayeredKeys('gallery')).toBe(getLayeredSettings().widgetValues.gallery);
    expect(getLayeredKeys('queue')).toBeUndefined();
  });
});
