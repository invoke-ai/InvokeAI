import { ButtonGroup, forwardRef } from '@chakra-ui/react';
import { memo } from 'react';

import type { InvButtonGroupProps } from './types';

export const InvButtonGroup = memo(
  forwardRef<InvButtonGroupProps, typeof ButtonGroup>(
    ({ isAttached = true, ...rest }: InvButtonGroupProps, ref) => {
      return <ButtonGroup ref={ref} isAttached={isAttached} {...rest} />;
    }
  )
);

InvButtonGroup.displayName = 'InvButtonGroup';
