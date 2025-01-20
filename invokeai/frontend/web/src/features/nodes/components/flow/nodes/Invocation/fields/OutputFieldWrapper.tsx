import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const sx = {
  position: 'relative',
  minH: 8,
  py: 0.5,
  alignItems: 'center',
  transitionProperty: 'opacity',
  transitionDuration: '0.1s',
  justifyContent: 'flex-end',
  '&[data-should-dim="true"]': {
    opacity: 0.5,
  },
} satisfies SystemStyleObject;

type OutputFieldWrapperProps = PropsWithChildren<{
  shouldDim: boolean;
}>;

export const OutputFieldWrapper = memo(({ shouldDim, children }: OutputFieldWrapperProps) => (
  <Flex sx={sx} data-should-dim={shouldDim}>
    {children}
  </Flex>
));

OutputFieldWrapper.displayName = 'OutputFieldWrapper';
