import type { WidgetViewProps } from '@workbench/types';

import { Flex, HStack, Spinner, Text } from '@chakra-ui/react';

export const WidgetLoadingFallback = ({ presentation }: { presentation: WidgetViewProps['presentation'] }) => {
  if (presentation === 'compact' || presentation === 'tooltip') {
    return (
      <HStack color="fg.subtle" gap="1.5" px="2">
        <Spinner size="xs" />
        <Text fontSize="2xs">Loading</Text>
      </HStack>
    );
  }

  return (
    <Flex align="center" color="fg.subtle" gap="2" h="full" justify="center" minH="8rem" w="full">
      <Spinner size="sm" />
      <Text fontSize="sm" fontWeight="600">
        Loading widget
      </Text>
    </Flex>
  );
};
