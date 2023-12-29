import { NumberInputField as ChakraNumberInputField } from '@chakra-ui/react';
import { useGlobalModifiersSetters } from 'common/hooks/useGlobalModifiers';
import type { KeyboardEventHandler } from 'react';
import { memo, useCallback } from 'react';

import type { InvNumberInputFieldProps } from './types';

export const InvNumberInputField = memo((props: InvNumberInputFieldProps) => {
  const { onKeyUp, onKeyDown, children, ...rest } = props;
  const { setShift } = useGlobalModifiersSetters();

  const _onKeyUp: KeyboardEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      onKeyUp?.(e);
      setShift(e.key === 'Shift');
    },
    [onKeyUp, setShift]
  );
  const _onKeyDown: KeyboardEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      onKeyDown?.(e);
      setShift(e.key === 'Shift');
    },
    [onKeyDown, setShift]
  );

  return (
    <ChakraNumberInputField onKeyUp={_onKeyUp} onKeyDown={_onKeyDown} {...rest}>
      {children}
    </ChakraNumberInputField>
  );
});

InvNumberInputField.displayName = 'InvNumberInputField';
