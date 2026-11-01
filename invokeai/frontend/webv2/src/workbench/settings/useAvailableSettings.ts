import { useCapabilities } from '@features/identity';
import { useMemo } from 'react';

import type { SettingsSection } from './catalog';

import { getAvailableSettings, settingsCatalog } from './catalog';

/** The settings this account can edit, the same on every surface (see `getAvailableSettings`). */
export const useAvailableSettings = (): SettingsSection[] => {
  const { canManageAppConfig } = useCapabilities();
  return useMemo(() => getAvailableSettings(settingsCatalog, { canManageAppConfig }), [canManageAppConfig]);
};
