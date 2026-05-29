import { Badge, Flex, FormControl, FormLabel, Text, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetExternalProviderStatusesQuery } from 'services/api/endpoints/appInfo';

import { getExternalProviderStatusBadgeInfo } from './externalProviderStatusUtils';

export const ExternalProviderStatusList = memo(() => {
  const { t } = useTranslation();
  const { data } = useGetExternalProviderStatusesQuery();

  if (!data || data.length === 0) {
    return null;
  }

  const sortedProviders = [...data].sort((a, b) => a.provider_id.localeCompare(b.provider_id));

  return (
    <FormControl>
      <FormLabel>{t('settings.externalProviders')}</FormLabel>
      <Flex direction="column" gap={2}>
        {sortedProviders.map((status) => {
          const badgeInfo = getExternalProviderStatusBadgeInfo(status);
          const tooltip = badgeInfo.tooltipMessage ?? (badgeInfo.tooltipKey ? t(badgeInfo.tooltipKey) : null);
          return (
            <Flex key={status.provider_id} alignItems="center" justifyContent="space-between" gap={3}>
              <Text fontWeight="semibold">{status.provider_id}</Text>
              <Tooltip label={tooltip} isDisabled={!tooltip}>
                <Badge colorScheme={badgeInfo.colorScheme}>{t(badgeInfo.labelKey)}</Badge>
              </Tooltip>
            </Flex>
          );
        })}
      </Flex>
    </FormControl>
  );
});

ExternalProviderStatusList.displayName = 'ExternalProviderStatusList';
