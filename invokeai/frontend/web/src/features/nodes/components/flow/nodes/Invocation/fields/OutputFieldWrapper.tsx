import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const sx = {
  position: 'relative',
  minH: 8,
  py: 0.5,
  alignItems: 'center',
  justifyContent: 'flex-end',
} satisfies SystemStyleObject;

export const OutputFieldWrapper = memo(({ children }: PropsWithChildren) => <Flex sx={sx}>{children}</Flex>);

OutputFieldWrapper.displayName = 'OutputFieldWrapper';
