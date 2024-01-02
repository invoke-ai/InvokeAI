import { Flex } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren;

const ProcessorWrapper = (props: Props) => {
  return (
    <Flex flexDir="column" gap={4}>
      {props.children}
    </Flex>
  );
};

export default memo(ProcessorWrapper);
