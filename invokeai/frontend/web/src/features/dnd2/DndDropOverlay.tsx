import { Flex, Text } from '@invoke-ai/ui-library';
import type { DndState } from 'features/dnd2/types';
import { memo } from 'react';

type Props = {
  dndState: DndState;
  label?: string;
  withBackdrop?: boolean;
};

export const DndDropOverlay = memo((props: Props) => {
  const { dndState, label, withBackdrop = true } = props;

  if (dndState === 'idle') {
    return null;
  }

  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0}>
      <Flex
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        bg={withBackdrop ? 'base.900' : 'transparent'}
        opacity={0.7}
        borderRadius="base"
        alignItems="center"
        justifyContent="center"
        transitionProperty="common"
        transitionDuration="0.1s"
      />

      <Flex
        position="absolute"
        top={0.5}
        right={0.5}
        bottom={0.5}
        left={0.5}
        opacity={1}
        borderWidth={1.5}
        borderColor={dndState === 'over' ? 'invokeYellow.300' : 'base.300'}
        borderRadius="base"
        borderStyle="dashed"
        transitionProperty="common"
        transitionDuration="0.1s"
        alignItems="center"
        justifyContent="center"
      >
        {label && (
          <Text
            fontSize="lg"
            fontWeight="semibold"
            color={dndState === 'over' ? 'invokeYellow.300' : 'base.300'}
            transitionProperty="common"
            transitionDuration="0.1s"
            textAlign="center"
          >
            {label}
          </Text>
        )}
      </Flex>
    </Flex>
  );
});

DndDropOverlay.displayName = 'DndDropOverlay';
