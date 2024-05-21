import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type InputFieldWrapperProps = PropsWithChildren<{
  shouldDim: boolean;
}>;

export const InputFieldWrapper = memo(({ shouldDim, children }: InputFieldWrapperProps) => {
  return (
    <Flex
      position="relative"
      minH={8}
      py={0.5}
      alignItems="center"
      opacity={shouldDim ? 0.5 : 1}
      transitionProperty="opacity"
      transitionDuration="0.1s"
      w="full"
      h="full"
    >
      {children}
    </Flex>
  );
});

InputFieldWrapper.displayName = 'InputFieldWrapper';
