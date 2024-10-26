import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

type Props = {
  isOver: boolean;
  label?: string;
  withBackdrop?: boolean;
};

const IAIDropOverlay = (props: Props) => {
  const { isOver, label, withBackdrop = true } = props;
  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0}>
      <Flex
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        w="full"
        h="full"
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
        borderColor={isOver ? 'invokeYellow.300' : 'base.500'}
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
            color={isOver ? 'invokeYellow.300' : 'base.500'}
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
};

export default memo(IAIDropOverlay);
