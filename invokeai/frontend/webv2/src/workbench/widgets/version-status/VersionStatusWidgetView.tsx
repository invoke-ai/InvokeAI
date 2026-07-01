import type { WidgetViewProps } from '@workbench/types';

import { Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { InfoIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const VersionStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          {t('widgets.versionStatus.label')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.versionStatus.description')}
        </Text>
      </Stack>
    );
  }

  return <StatusWidgetChip icon={InfoIcon}>{t('widgets.versionStatus.chipLabel')}</StatusWidgetChip>;
};
