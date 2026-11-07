import { HStack } from '@chakra-ui/react';
import { Group } from '@platform/ui/Group';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';

import { InvokeButton } from './InvokeButton';
import { IterationsField } from './IterationsField';
import { useInvocationState } from './useInvocationState';

// Keep the controls mounted and focused while changing their visual order. Group's default corner rules follow DOM order.
const LEFT_ORDER_CSS = {
  '&& > :first-child': {
    order: 1,
    borderStartStartRadius: '0 !important',
    borderEndStartRadius: '0 !important',
    borderStartEndRadius: 'control !important',
    borderEndEndRadius: 'control !important',
    marginStart: '-1px',
    marginEnd: '0',
  },
  '& > :nth-child(2)': { order: 0 },
} as const;

export const InvocationCluster = () => {
  const state = useInvocationState();
  const invokeOnLeft = useWorkbenchPreferenceSelector((preferences) => preferences.showInvokeButtonOnLeft);

  return (
    <HStack flexShrink={0} gap="2" justify="end">
      <Group attached css={invokeOnLeft ? LEFT_ORDER_CSS : undefined}>
        <IterationsField invokeOnLeft={invokeOnLeft} />
        <InvokeButton state={state} />
      </Group>
    </HStack>
  );
};
