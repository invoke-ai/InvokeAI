import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const sx = {
  position: 'relative',
  minH: 8,
  py: 0.5,
  alignItems: 'center',
  w: 'full',
  h: 'full',
} satisfies SystemStyleObject;

export const InputFieldWrapper = memo(({ children }: PropsWithChildren) => {
  return <Flex sx={sx}>{children}</Flex>;
});

InputFieldWrapper.displayName = 'InputFieldWrapper';
