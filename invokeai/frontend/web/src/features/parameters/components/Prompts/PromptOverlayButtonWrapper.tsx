import { Flex } from '@invoke-ai/ui-library';
import { memo, type PropsWithChildren } from 'react';

export const PromptOverlayButtonWrapper = memo((props: PropsWithChildren) => (
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
));

PromptOverlayButtonWrapper.displayName = 'PromptOverlayButtonWrapper';
