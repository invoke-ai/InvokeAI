import type { ReactNode } from 'react';

import { HStack, Text } from '@chakra-ui/react';

export const PANEL_HEADER_CONTROL_HEIGHT = '6';

export const PromptPanelHeader = ({ children, label }: { label: string; children?: ReactNode }) => (
  <HStack justify="space-between" minH={PANEL_HEADER_CONTROL_HEIGHT}>
    <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
      {label}
    </Text>
    {children}
  </HStack>
);
