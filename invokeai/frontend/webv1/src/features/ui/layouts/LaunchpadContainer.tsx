import { Flex, Heading } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const LaunchpadContainer = memo((props: PropsWithChildren<{ heading: string }>) => {
  return (
    <Flex flexDir="column" h="full" w="full" alignItems="center" justifyContent="center" gap={2}>
      <Flex flexDir="column" w="full" gap={4} px={14} maxW={768}>
        <Heading>{props.heading}</Heading>
        <Flex flexDir="column" gap={4}>
          {props.children}
        </Flex>
      </Flex>
    </Flex>
  );
});
LaunchpadContainer.displayName = 'LaunchpadContainer';
