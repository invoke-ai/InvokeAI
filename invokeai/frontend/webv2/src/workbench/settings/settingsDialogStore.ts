import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { SettingsDestination, SettingsSectionId } from './contracts';

import { settingsDialogResource } from './dialogResource';

interface SettingsDialogSnapshot extends SettingsDestination {
  isOpen: boolean;
  query: string;
  searchSection: string | null;
  returnFocus: HTMLElement | null;
  generation: number;
}
const initialSnapshot = (): SettingsDialogSnapshot => ({
  isOpen: false,
  sectionId: 'appearance',
  query: '',
  searchSection: null,
  returnFocus: null,
  generation: 0,
});
export const settingsDialogStore = createExternalStore<SettingsDialogSnapshot>(initialSnapshot());
const sectionScrollPositions = new Map<SettingsSectionId, number>();
export const getSettingsSectionScroll = (sectionId: SettingsSectionId): number =>
  sectionScrollPositions.get(sectionId) ?? 0;
export const rememberSettingsSectionScroll = (sectionId: SettingsSectionId, offset: number): void => {
  sectionScrollPositions.set(sectionId, offset);
};
registerAccountOwnedResource({
  name: 'settings-dialog',
  clear: () => {
    sectionScrollPositions.clear();
    settingsDialogStore.setSnapshot({
      ...initialSnapshot(),
      generation: settingsDialogStore.getSnapshot().generation + 1,
    });
  },
});

export const openWorkbenchSettings = (
  destination?: SettingsSectionId | SettingsDestination,
  returnFocus?: HTMLElement
): void => {
  settingsDialogResource.preload();
  const current = settingsDialogStore.getSnapshot();
  const next = typeof destination === 'string' ? { sectionId: destination } : destination;
  settingsDialogStore.setSnapshot({
    ...current,
    isOpen: true,
    query: '',
    searchSection: null,
    entryId: undefined,
    target: undefined,
    ...next,
    returnFocus:
      returnFocus ??
      (typeof document !== 'undefined' && document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null),
  });
};
export const closeWorkbenchSettings = (): void => {
  settingsDialogStore.patchSnapshot({ isOpen: false });
};
export const setWorkbenchSettingsSection = (sectionId: SettingsSectionId, entryId?: string): void => {
  settingsDialogStore.patchSnapshot({
    sectionId,
    entryId,
    query: '',
    searchSection: null,
    target:
      settingsDialogStore.getSnapshot().sectionId === sectionId ? settingsDialogStore.getSnapshot().target : undefined,
  });
};
export const setSettingsQuery = (query: string): void => {
  settingsDialogStore.patchSnapshot({ query, searchSection: null, entryId: undefined });
};
