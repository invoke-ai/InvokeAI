import { HStack } from '@chakra-ui/react';
import { Group } from '@platform/ui/Group';

import { InvokeButton } from './InvokeButton';
import { IterationsField } from './IterationsField';
import { QueueCluster } from './QueueCluster';
import { RoutingControl } from './RoutingControl';
import { useInvocationState } from './useInvocationState';

/**
 * The right zone: two grouped controls.
 *
 * The split is semantic, not decorative. The first group is everything about
 * *dispatching* work — how many, run it, where from and to. The second is
 * everything about work already *dispatched*. Reading the bar left to right
 * therefore reads the lifecycle in order, and progress never lands on a control
 * whose job is to start something new.
 */
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
