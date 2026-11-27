import type { SettingDefinition } from '@platform/ui/settings/contracts';

import { Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useWorkbenchSettingsSelector } from './store';

export const SettingsScopeLabel = ({ scope }: { scope: SettingDefinition['scope'] }) => {
  const { t } = useTranslation();
  const preferenceScope = useWorkbenchSettingsSelector((snapshot) => snapshot.scope);
  // Only where it is news: a server-wide setting, or a preference kept for this install while nobody is signed in.
  const label =
    scope === 'server' ? 'server' : scope === 'preference' && preferenceScope === 'global' ? 'install' : null;
  if (!label) {
    return null;
  }
  return (
    <Text fontSize="xs" color="fg.muted">
      {t(`settingsDialog.scope.${label}`)}
    </Text>
  );
};
