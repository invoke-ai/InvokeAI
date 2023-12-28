import { forwardRef, Input } from '@chakra-ui/react';
import { useGlobalModifiersSetters } from 'common/hooks/useGlobalModifiers';
import { stopPastePropagation } from 'common/util/stopPastePropagation';
import type { KeyboardEvent } from 'react';
import { useCallback } from 'react';

import type { InvInputProps } from './types';

export const InvInput = forwardRef<InvInputProps, typeof Input>(
  (props: InvInputProps, ref) => {
    const { setShift } = useGlobalModifiersSetters();
    const onKeyUpDown = useCallback(
      (e: KeyboardEvent<HTMLInputElement>) => {
        setShift(e.shiftKey);
      },
      [setShift]
    );
    return (
      <Input
        ref={ref}
        onPaste={stopPastePropagation}
        onKeyUp={onKeyUpDown}
        onKeyDown={onKeyUpDown}
        {...props}
      />
    );
  }
);
