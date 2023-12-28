import { ButtonGroup, forwardRef } from '@chakra-ui/react';

import type { InvButtonGroupProps } from './types';

export const InvButtonGroup = forwardRef<
  InvButtonGroupProps,
  typeof ButtonGroup
>(({ isAttached = true, ...rest }: InvButtonGroupProps, ref) => {
  return <ButtonGroup ref={ref} isAttached={isAttached} {...rest} />;
});
