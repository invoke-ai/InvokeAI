import type { ChakraProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  onClick?: () => void;
  borderColor: ChakraProps['bg'];
}>;

export const LayerWrapper = memo(({ onClick, borderColor, children }: Props) => {
  return (
    <Flex gap={2} onClick={onClick} bg={borderColor} px={2} borderRadius="base" py="1px">
      <Flex flexDir="column" w="full" bg="base.850" borderRadius="base">
        {children}
      </Flex>
    </Flex>
  );
});

LayerWrapper.displayName = 'LayerWrapper';
