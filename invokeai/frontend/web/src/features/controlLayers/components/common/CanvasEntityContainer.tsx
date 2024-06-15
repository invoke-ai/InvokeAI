import type { ChakraProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

type Props = PropsWithChildren<{
  isSelected: boolean;
  onSelect: () => void;
  selectedBorderColor?: ChakraProps['bg'];
}>;

export const CanvasEntityContainer = memo(({ isSelected, onSelect, selectedBorderColor, children }: Props) => {
  const bg = useMemo(() => {
    if (isSelected) {
      return selectedBorderColor ?? 'base.400';
    }
    return 'base.800';
  }, [isSelected, selectedBorderColor]);
  return (
    <Flex
      position="relative"
      gap={2}
      onClick={onSelect}
      bg={bg}
      px={2}
      borderRadius="base"
      py="1px"
      transitionProperty="all"
      transitionDuration="0.2s"
    >
      <Flex flexDir="column" w="full" bg="base.850" borderRadius="base">
        {children}
      </Flex>
    </Flex>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
