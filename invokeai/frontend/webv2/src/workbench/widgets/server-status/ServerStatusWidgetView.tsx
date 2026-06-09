import { Stack, Text } from '@chakra-ui/react';
import { PiPlugsConnectedBold } from 'react-icons/pi';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import type { WidgetViewProps } from '../../types';

export const ServerStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          Server Status
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          Connected to Server. Detailed socket and API health will render here once backend status data is connected.
        </Text>
      </Stack>
    );
  }

  return <StatusWidgetChip icon={PiPlugsConnectedBold}>Connected to Server</StatusWidgetChip>;
};
