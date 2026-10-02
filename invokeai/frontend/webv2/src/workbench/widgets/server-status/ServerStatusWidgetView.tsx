import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CircleXIcon, PlugZapIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const ServerStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const backendConnection = useWorkbenchSelector((snapshot) => snapshot.backendConnection);
  const isConnected = backendConnection.status === 'connected';
  const isDisconnected = backendConnection.status === 'disconnected';
  const label = isConnected
    ? t('widgets.serverStatus.connected')
    : isDisconnected
      ? t('widgets.serverStatus.disconnected')
      : t('widgets.serverStatus.connecting');

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          {t('widgets.serverStatus.label')}
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {backendConnection.error
            ? t('widgets.serverStatus.labelWithError', { error: backendConnection.error, label })
            : label}
        </Text>
      </Stack>
    );
  }

  if (isDisconnected) {
    return (
      <StatusWidgetChip icon={CircleXIcon} tone="error">
        {label}
      </StatusWidgetChip>
    );
  }

  return <StatusWidgetChip icon={PlugZapIcon}>{label}</StatusWidgetChip>;
};
