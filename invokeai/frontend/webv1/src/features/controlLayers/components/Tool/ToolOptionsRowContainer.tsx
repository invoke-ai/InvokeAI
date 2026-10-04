import { Flex, type FlexProps } from '@invoke-ai/ui-library';
import { forwardRef } from 'react';

export const ToolOptionsRowContainer = forwardRef<HTMLDivElement, FlexProps>((props, ref) => {
  return (
    <Flex
      ref={ref}
      alignItems="center"
      h="full"
      flexGrow={1}
      flexShrink={1}
      justifyContent="flex-start"
      px={4}
      w="full"
      minW={0}
      {...props}
    />
  );
});

ToolOptionsRowContainer.displayName = 'ToolOptionsRowContainer';
