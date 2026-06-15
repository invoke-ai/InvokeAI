import type { ReactNode, Ref, RefObject } from 'react';

import { Tooltip as ChakraTooltip, Portal } from '@chakra-ui/react';

export interface TooltipProps extends ChakraTooltip.RootProps {
  showArrow?: boolean;
  portalled?: boolean;
  portalRef?: RefObject<HTMLElement | null>;
  content: ReactNode;
  contentProps?: ChakraTooltip.ContentProps;
  disabled?: boolean;
  ref?: Ref<HTMLDivElement>;
}

/**
 * Workbench tooltip. Chrome comes from the `tooltip` slot-recipe override in
 * `theme/recipes.ts`, so this wrapper only provides the trigger/portal
 * structure and the `content` convenience API.
 */
export const Tooltip = (props: TooltipProps) => {
  const { showArrow, children, disabled, portalled = true, content, contentProps, portalRef, ref, ...rest } = props;

  if (disabled) {
    return children;
  }

  return (
    <ChakraTooltip.Root {...rest}>
      <ChakraTooltip.Trigger asChild>{children}</ChakraTooltip.Trigger>
      <Portal disabled={!portalled} container={portalRef}>
        <ChakraTooltip.Positioner>
          <ChakraTooltip.Content ref={ref} {...contentProps}>
            {showArrow && (
              <ChakraTooltip.Arrow>
                <ChakraTooltip.ArrowTip />
              </ChakraTooltip.Arrow>
            )}
            {content}
          </ChakraTooltip.Content>
        </ChakraTooltip.Positioner>
      </Portal>
    </ChakraTooltip.Root>
  );
};
