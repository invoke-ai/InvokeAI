import type { ChakraProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo, useCallback } from 'react';

type Props = PropsWithChildren<{
  isSelected: boolean;
  onSelect: () => void;
  selectedBorderColor?: ChakraProps['bg'];
}>;

export const CanvasEntityContainer = memo((props: Props) => {
  const { isSelected, onSelect, selectedBorderColor = 'base.400', children } = props;
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
      bg={isSelected ? 'base.800' : 'base.850'}
      onClick={_onSelect}
      borderInlineStartWidth={5}
      borderColor={isSelected ? selectedBorderColor : 'base.800'}
      borderRadius="base"
    >
      {children}
    </Flex>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';
