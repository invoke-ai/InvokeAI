import {
  Flex,
  FormControl as ChakraFormControl,
  FormHelperText as ChakraFormHelperText,
  forwardRef,
} from '@chakra-ui/react';
import { InvControlGroupContext } from 'common/components/InvControl/InvControlGroup';
import { memo, useContext } from 'react';

import { InvLabel } from './InvLabel';
import type { InvControlProps } from './types';

export const InvControl = memo(
  forwardRef<InvControlProps, typeof ChakraFormControl>(
    (props: InvControlProps, ref) => {
      const {
        children,
        helperText,
        feature,
        orientation,
        renderInfoPopoverInPortal = true,
        isDisabled,
        labelProps,
        label,
        ...formControlProps
      } = props;

      const ctx = useContext(InvControlGroupContext);

      if (helperText) {
        return (
          <ChakraFormControl
            ref={ref}
            variant="withHelperText"
            orientation={orientation ?? ctx.orientation}
            isDisabled={isDisabled ?? ctx.isDisabled}
            {...formControlProps}
          >
            <Flex>
              {label && (
                <InvLabel
                  feature={feature}
                  renderInfoPopoverInPortal={renderInfoPopoverInPortal}
                  {...labelProps}
                >
                  {label}
                </InvLabel>
              )}
              {children}
            </Flex>
            <ChakraFormHelperText>{helperText}</ChakraFormHelperText>
          </ChakraFormControl>
        );
      }

      return (
        <ChakraFormControl
          ref={ref}
          isDisabled={isDisabled ?? ctx.isDisabled}
          orientation={orientation ?? ctx.orientation}
          {...formControlProps}
        >
          {label && (
            <InvLabel
              feature={feature}
              renderInfoPopoverInPortal={renderInfoPopoverInPortal}
              {...labelProps}
            >
              {label}
            </InvLabel>
          )}
          {children}
        </ChakraFormControl>
      );
    }
  )
);

InvControl.displayName = 'InvControl';
