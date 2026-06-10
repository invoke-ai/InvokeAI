import { Tooltip as ChakraTooltip, Portal, useSlotRecipe } from '@chakra-ui/react';
import * as React from 'react';

import { workbenchTooltipRecipe } from '../../../theme/recipes';

export interface TooltipProps extends ChakraTooltip.RootProps {
  showArrow?: boolean;
  portalled?: boolean;
  portalRef?: React.RefObject<HTMLElement | null>;
  content: React.ReactNode;
  contentProps?: ChakraTooltip.ContentProps;
  disabled?: boolean;
}

export const Tooltip = React.forwardRef<HTMLDivElement, TooltipProps>(function Tooltip(props, ref) {
  const { showArrow, children, disabled, portalled = true, content, contentProps, portalRef, ...rest } = props;
  const recipe = useSlotRecipe({ recipe: workbenchTooltipRecipe });
  const styles = recipe();
  const { css: contentCss, ...restContentProps } = contentProps ?? {};

  if (disabled) {
    return children;
  }

  return (
    <ChakraTooltip.Root {...rest}>
      <ChakraTooltip.Trigger asChild>{children}</ChakraTooltip.Trigger>
      <Portal disabled={!portalled} container={portalRef}>
        <ChakraTooltip.Positioner>
          <ChakraTooltip.Content ref={ref} css={[styles.content, contentCss]} {...restContentProps}>
            {showArrow && (
              <ChakraTooltip.Arrow css={styles.arrow}>
                <ChakraTooltip.ArrowTip css={styles.arrowTip} />
              </ChakraTooltip.Arrow>
            )}
            {content}
          </ChakraTooltip.Content>
        </ChakraTooltip.Positioner>
      </Portal>
    </ChakraTooltip.Root>
  );
});
