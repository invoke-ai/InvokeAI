import { Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const FetchingModelsLoader = memo(({ loadingMessage }: { loadingMessage?: string }) => {
  return (
    <Flex flexDirection="column" gap={4} borderRadius="base" p={4} bg="base.800">
      <Flex justifyContent="center" alignItems="center" flexDirection="column" p={4} gap={8}>
        <Spinner />
        <Text variant="subtext">{loadingMessage ? loadingMessage : 'Fetching...'}</Text>
      </Flex>
    </Flex>
  );
});

FetchingModelsLoader.displayName = 'FetchingModelsLoader';
