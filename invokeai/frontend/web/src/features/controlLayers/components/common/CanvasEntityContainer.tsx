import type { ChakraProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';

type Props = PropsWithChildren<{
  isSelected: boolean;
  onSelect: () => void;
  selectedBorderColor?: ChakraProps['bg'];
}>;

export const CanvasEntityContainer = memo(({ isSelected, onSelect, selectedBorderColor, children }: Props) => {
  const borderColor = useMemo(() => {
    if (isSelected) {
      return selectedBorderColor ?? 'base.400';
    }
    return 'base.800';
  }, [isSelected, selectedBorderColor]);
  const _onSelect = useCallback(() => {
    if (isSelected) {
      return;
    }
    onSelect();
  }, [isSelected, onSelect]);

  return (
    <Flex
      position="relative" // necessary for drop overlay
      flexDir="column"
      w="full"
      bg="base.850"
      onClick={_onSelect}
      borderInlineStartWidth={5}
      borderColor={borderColor}
      opacity={isSelected ? 1 : 0.6}
      borderRadius="base"
      transitionProperty="all"
      transitionDuration="0.2s"
    >
      {children}
    </Flex>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
