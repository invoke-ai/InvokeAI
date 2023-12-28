import { Flex } from '@chakra-ui/layout';
import type { PropsWithChildren } from 'react';

export const PromptOverlayButtonWrapper = (props: PropsWithChildren) => (
  <Flex
    pos="absolute"
    insetBlockStart={0}
    insetInlineEnd={0}
    flexDir="column"
    p={2}
    gap={2}
    alignItems="center"
    justifyContent="center"
  >
    {props.children}
  </Flex>
);
