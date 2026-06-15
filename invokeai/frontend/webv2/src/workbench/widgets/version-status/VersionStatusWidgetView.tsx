import { Stack, Text } from '@chakra-ui/react';
import { InfoIcon } from 'lucide-react';

import { StatusWidgetChip } from '@workbench/components/WidgetFrames';
import type { WidgetViewProps } from '@workbench/types';

export const VersionStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          Version
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          Invoke V7 shell version 7.0.
        </Text>
      </Stack>
    );
  }

  return <StatusWidgetChip icon={InfoIcon}>Version 7.0</StatusWidgetChip>;
};
