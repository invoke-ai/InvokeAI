import { HStack, Icon, Link, Spinner, Stack, Text } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { DiscordIcon, GithubIcon } from '@platform/ui/VendoredIcon';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { refreshAboutInfo, useAboutInfo } from './aboutInfoStore';

const GITHUB_URL = 'https://github.com/invoke-ai/InvokeAI';
const DISCORD_URL = 'https://discord.gg/ZmtBAhwWhy';

export const AboutSettings = () => {
  const { t } = useTranslation();
  const { canManageAppConfig } = useCapabilities();
  const info = useAboutInfo();

  useMountEffect(() => {
    void refreshAboutInfo(canManageAppConfig);
  });

  const systemInfo = useMemo(
    () => ({
      version: info.version,
      dependencies: info.dependencies,
      ...(info.runtimeConfig ? { config: info.runtimeConfig } : {}),
    }),
    [info.dependencies, info.runtimeConfig, info.version]
  );

  return (
    <Stack gap="4">
      <HStack gap="4">
        <Text fontSize="xs" fontWeight="700">
          {info.version ? `Invoke v${info.version}` : 'Invoke'}
        </Text>
        <HStack gap="3">
          <Link fontSize="xs" href={GITHUB_URL} rel="noreferrer" target="_blank">
            <Icon as={GithubIcon} boxSize="3.5" />
            {t('settings.about.github')}
          </Link>
          <Link fontSize="xs" href={DISCORD_URL} rel="noreferrer" target="_blank">
            <Icon as={DiscordIcon} boxSize="3.5" />
            {t('settings.about.discord')}
          </Link>
        </HStack>
      </HStack>

      {info.loadState === 'loading' || info.loadState === 'idle' ? (
        <HStack color="fg.muted" gap="2">
          <Spinner size="xs" />
          <Text fontSize="xs">{t('settings.about.loading')}</Text>
        </HStack>
      ) : info.loadState === 'error' ? (
        <Text color="fg.error" fontSize="xs">
          {info.error}
        </Text>
      ) : (
        <JsonPreview label={t('settings.about.systemInformation')} maxH="24rem" value={systemInfo} />
      )}
    </Stack>
  );
};
