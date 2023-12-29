import { Flex } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';

type Props = PropsWithChildren;

export default function ProcessorWrapper(props: Props) {
  return (
    <Flex flexDir="column" gap={2} pb={2}>
      {props.children}
    </Flex>
  );
}
