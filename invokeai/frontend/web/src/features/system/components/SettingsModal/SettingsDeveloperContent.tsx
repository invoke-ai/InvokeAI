import { Flex } from '@invoke-ai/ui-library';
import { SettingsDeveloperLogIsEnabled } from 'features/system/components/SettingsModal/SettingsDeveloperLogIsEnabled';
import { SettingsDeveloperLogLevel } from 'features/system/components/SettingsModal/SettingsDeveloperLogLevel';
import { SettingsDeveloperLogNamespaces } from 'features/system/components/SettingsModal/SettingsDeveloperLogNamespaces';
import { memo } from 'react';

export const SettingsDeveloperContent = memo(() => {
  return (
    <Flex flexDir="column" gap={4}>
      <SettingsDeveloperLogIsEnabled />
      <SettingsDeveloperLogLevel />
      <SettingsDeveloperLogNamespaces />
    </Flex>
  );
});

SettingsDeveloperContent.displayName = 'SettingsDeveloperContent';
