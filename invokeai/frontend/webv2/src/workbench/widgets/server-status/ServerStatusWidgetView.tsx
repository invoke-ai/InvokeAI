import type { WidgetViewProps } from '@workbench/types';

import { Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CircleXIcon, PlugZapIcon } from 'lucide-react';

export const ServerStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const backendConnection = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection);
  const isConnected = backendConnection.status === 'connected';
  const isDisconnected = backendConnection.status === 'disconnected';
  const label = isConnected
    ? 'Connected to Server'
    : isDisconnected
      ? 'Disconnected from Server'
      : 'Connecting to Server';

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          Server Status
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          {label}
          {backendConnection.error ? `: ${backendConnection.error}` : ''}
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
