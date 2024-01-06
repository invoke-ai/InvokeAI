import { Flex, FormLabel, forwardRef } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import { InvControlGroupContext } from 'common/components/InvControl/InvControlGroup';
import { memo, useContext } from 'react';

import type { InvLabelProps } from './types';

export const InvLabel = memo(
  forwardRef<InvLabelProps, typeof FormLabel>(
    (
      { feature, renderInfoPopoverInPortal, children, ...rest }: InvLabelProps,
      ref
    ) => {
      const shouldEnableInformationalPopovers = useAppSelector(
        (s) => s.system.shouldEnableInformationalPopovers
      );

      const ctx = useContext(InvControlGroupContext);
      if (feature && shouldEnableInformationalPopovers) {
        return (
          <IAIInformationalPopover
            feature={feature}
            inPortal={renderInfoPopoverInPortal}
          >
            <Flex as="span">
              <FormLabel ref={ref} {...ctx.labelProps} {...rest}>
                {children}
              </FormLabel>
            </Flex>
          </IAIInformationalPopover>
        );
      }
      return (
        <FormLabel ref={ref} {...ctx.labelProps} {...rest}>
          {children}
        </FormLabel>
      );
    }
  )
);

InvLabel.displayName = 'InvLabel';
