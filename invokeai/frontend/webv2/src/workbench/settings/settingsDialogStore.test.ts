import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  closeWorkbenchSettings,
  getSettingsSectionScroll,
  openWorkbenchSettings,
  rememberSettingsSectionScroll,
  setSettingsQuery,
  setWorkbenchSettingsSection,
  settingsDialogStore,
} from './settingsDialogStore';

beforeEach(() => accountLifecycle.activate('settings-navigation-test'));

describe('settings navigation', () => {
  it('opens a specific widget instance and field, clearing stale search state', () => {
    openWorkbenchSettings('appearance');
    setSettingsQuery('motion');
    settingsDialogStore.patchSnapshot({ searchSection: 'appearance' });
    const target = { instanceId: 'canvas-instance', projectId: 'project-a' };

    openWorkbenchSettings({ entryId: 'showGrid', sectionId: 'canvas', target });

    expect(settingsDialogStore.getSnapshot()).toMatchObject({
      entryId: 'showGrid',
      isOpen: true,
      query: '',
      searchSection: null,
      sectionId: 'canvas',
      target,
    });
  });

  it('remembers the browsed section after closing but releases an explicit target when reopened generically', () => {
    openWorkbenchSettings({
      entryId: 'showGrid',
      sectionId: 'canvas',
      target: { projectId: 'project-a', instanceId: 'canvas-instance' },
    });
    closeWorkbenchSettings();
    expect(settingsDialogStore.getSnapshot().isOpen).toBe(false);
    openWorkbenchSettings();
    expect(settingsDialogStore.getSnapshot()).toMatchObject({
      entryId: undefined,
      isOpen: true,
      sectionId: 'canvas',
      target: undefined,
    });
  });

  it('clears section filtering while editing the query without changing the browsing destination', () => {
    openWorkbenchSettings('canvas');
    setSettingsQuery('grid');
    rememberSettingsSectionScroll('canvas', 320);
    settingsDialogStore.patchSnapshot({ searchSection: 'canvas' });
    setSettingsQuery('');
    expect(settingsDialogStore.getSnapshot()).toMatchObject({ query: '', searchSection: null, sectionId: 'canvas' });
  });

  it('reveals an entry in its section and removes the prior widget target and search', () => {
    openWorkbenchSettings({ sectionId: 'canvas', target: { projectId: 'project-a', instanceId: 'canvas-instance' } });
    setSettingsQuery('filmstrip');
    setWorkbenchSettingsSection('preview', 'filmstripVisible');
    expect(settingsDialogStore.getSnapshot()).toMatchObject({
      entryId: 'filmstripVisible',
      query: '',
      searchSection: null,
      sectionId: 'preview',
      target: undefined,
    });
  });

  it('clears destinations and search synchronously on account changes, invalidating the prior dialog generation', () => {
    openWorkbenchSettings({
      sectionId: 'canvas',
      entryId: 'showGrid',
      target: { projectId: 'project-a', instanceId: 'canvas-instance' },
    });
    setSettingsQuery('grid');
    const generation = settingsDialogStore.getSnapshot().generation;
    accountLifecycle.activate('other-user');

    expect(getSettingsSectionScroll('canvas')).toBe(0);
    expect(settingsDialogStore.getSnapshot()).toEqual({
      generation: generation + 1,
      isOpen: false,
      query: '',
      returnFocus: null,
      searchSection: null,
      sectionId: 'appearance',
    });
    openWorkbenchSettings();
    expect(settingsDialogStore.getSnapshot().sectionId).toBe('appearance');
  });
});
