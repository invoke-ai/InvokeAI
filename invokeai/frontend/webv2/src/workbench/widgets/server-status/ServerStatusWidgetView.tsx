import { Stack, Text } from '@chakra-ui/react';
import { PiPlugsConnectedBold, PiXCircleBold } from 'react-icons/pi';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

export const ServerStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { state } = useWorkbench();
  const { backendConnection } = state;
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
      <StatusWidgetChip borderColor="fg.error" color="fg.error" icon={PiXCircleBold}>
        {label}
      </StatusWidgetChip>
    );
  }

  return <StatusWidgetChip icon={PiPlugsConnectedBold}>{label}</StatusWidgetChip>;
};
