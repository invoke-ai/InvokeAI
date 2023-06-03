import { Flex } from '@chakra-ui/react';
import { PropsWithChildren } from 'react';

type Props = PropsWithChildren;

export default function ProcessorWrapper(props: Props) {
  return <Flex sx={{ flexDirection: 'column', gap: 2 }}>{props.children}</Flex>;
}
