import {
  Flex,
  FormControl as ChakraFormControl,
  FormErrorMessage as ChakraFormErrorMessage,
  FormHelperText as ChakraFormHelperText,
  forwardRef,
} from '@chakra-ui/react';
import { InvControlGroupContext } from 'common/components/InvControl/InvControlGroup';
import { memo, useContext, useMemo } from 'react';

import { InvLabel } from './InvLabel';
import type { InvControlProps } from './types';

export const InvControl = memo(
  forwardRef<InvControlProps, typeof ChakraFormControl>(
    (props: InvControlProps, ref) => {
      const {
        children,
        helperText,
        feature,
        orientation: _orientation,
        renderInfoPopoverInPortal = true,
        isDisabled: _isDisabled,
        labelProps,
        label,
        error,
        ...formControlProps
      } = props;

      const ctx = useContext(InvControlGroupContext);

      const orientation = useMemo(
        () => _orientation ?? ctx.orientation,
        [_orientation, ctx.orientation]
      );

      const isDisabled = useMemo(
        () => _isDisabled ?? ctx.isDisabled,
        [_isDisabled, ctx.isDisabled]
      );

      return (
        <ChakraFormControl
          ref={ref}
          orientation={orientation}
          isDisabled={isDisabled}
          {...formControlProps}
          {...ctx.controlProps}
        >
          <Flex className="invcontrol-label-wrapper">
            {label && (
              <InvLabel
                feature={feature}
                renderInfoPopoverInPortal={renderInfoPopoverInPortal}
                {...labelProps}
              >
                {label}
              </InvLabel>
            )}
            <Flex className="invcontrol-input-wrapper">{children}</Flex>
          </Flex>
          {helperText && (
            <ChakraFormHelperText>{helperText}</ChakraFormHelperText>
          )}
          {error && <ChakraFormErrorMessage>{error}</ChakraFormErrorMessage>}
        </ChakraFormControl>
      );
    }
  )
);

InvControl.displayName = 'InvControl';
