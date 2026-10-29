import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import { useCapabilities } from '@features/identity';
import { useOptionalWorkbenchSelector } from '@workbench/WorkbenchContext';

import type { SettingsSection } from './catalog';

import { getAvailableSettings, settingsCatalog } from './catalog';

/** The open project's widget types as one stable string, so the selection only changes when the set does. */
const selectWidgetTypeKey = (snapshot: WorkbenchSnapshot): string =>
  [...new Set(Object.values(snapshot.activeProject.widgetInstances).map((instance) => instance.typeId))]
    .sort()
    .join(' ');

/**
 * Settings editable where this renders: outside a workbench (the Launchpad) that leaves out project and widget
 * settings, except that `forSearch` keeps project settings so a search can say they need an open project.
 */
export const useAvailableSettings = ({ forSearch = false }: { forSearch?: boolean } = {}): SettingsSection[] => {
  const { canManageAppConfig } = useCapabilities();
  const widgetTypeKey = useOptionalWorkbenchSelector<string | null>(selectWidgetTypeKey, null);
  return getAvailableSettings(settingsCatalog, {
    canManageAppConfig,
    forSearch,
    widgetTypeIds: widgetTypeKey === null ? null : new Set(widgetTypeKey.split(' ')),
  });
};
