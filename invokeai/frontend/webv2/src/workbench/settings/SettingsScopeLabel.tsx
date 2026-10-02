import type { SettingDefinition } from '@platform/ui/settings/contracts';

import { Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { useWorkbenchSettingsSelector } from './store';

export const SettingsScopeLabel = ({ scope }: { scope: SettingDefinition['scope'] }) => {
  const { t } = useTranslation();
  const preferenceScope = useWorkbenchSettingsSelector((snapshot) => snapshot.scope);
  if (scope === 'none') {
    return null;
  }
  const label = scope === 'preference' && preferenceScope === 'global' ? 'install' : scope;
  return (
    <Text fontSize="2xs" color="fg.muted">
      {t(`settingsDialog.scope.${label}`)}
    </Text>
  );
};
