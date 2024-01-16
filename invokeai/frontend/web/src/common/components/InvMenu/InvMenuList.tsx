import {
  forwardRef,
  MenuList as ChakraMenuList,
  Portal,
} from '@chakra-ui/react';
import { skipMouseEvent } from 'common/util/skipMouseEvent';
import { memo } from 'react';

import { menuListMotionProps } from './constants';
import type { InvMenuListProps } from './types';

export const InvMenuList = memo(
  forwardRef<InvMenuListProps, typeof ChakraMenuList>(
    (props: InvMenuListProps, ref) => {
      return (
        <Portal>
          <ChakraMenuList
            ref={ref}
            motionProps={menuListMotionProps}
            onContextMenu={skipMouseEvent}
            {...props}
          />
        </Portal>
      );
    }
  )
);

InvMenuList.displayName = 'InvMenuList';
