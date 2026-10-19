import { HStack } from '@chakra-ui/react';
import { Group } from '@platform/ui/Group';

import { InvokeButton } from './InvokeButton';
import { IterationsField } from './IterationsField';
import { QueueCluster } from './QueueCluster';
import { RoutingControl } from './RoutingControl';
import { useInvocationState } from './useInvocationState';

export const InvocationCluster = () => {
  const state = useInvocationState();

  return (
    <HStack flexShrink={0} gap="2" justify="end">
      <Group attached>
        <IterationsField />
        <InvokeButton state={state} />
        <RoutingControl state={state} />
      </Group>
      <QueueCluster />
    </HStack>
  );
};
