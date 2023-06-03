import { Flex } from '@chakra-ui/react';
import { PropsWithChildren } from 'react';

type ProcessorOptionsContainerProps = PropsWithChildren;

export default function ProcessorOptionsContainer(
  props: ProcessorOptionsContainerProps
) {
  return (
    <Flex sx={{ flexDirection: 'column', gap: 2, p: 2 }}>{props.children}</Flex>
  );
}
