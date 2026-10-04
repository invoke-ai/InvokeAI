import type { ComponentProps } from 'react';

import { Popover, usePopoverContext } from '@chakra-ui/react';

import { useRegisterWidgetOverlay } from './widgetOverlays';

type PopoverContentProps = ComponentProps<typeof Popover.Content> & {
  /** Anchored panels keep the arrow; dropdown-shaped popovers (pickers) opt out. */
  showArrow?: boolean;
};

/** Popover content closes with its owning widget; the theme recipe owns chrome. */
export const PopoverContent = ({ children, showArrow = true, ...props }: PopoverContentProps) => {
  const popover = usePopoverContext();
  const stale = useRegisterWidgetOverlay(popover.open, popover.setOpen);
  if (stale) {
    return null;
  }
  return (
    <Popover.Content {...props}>
      {showArrow ? (
        <Popover.Arrow>
          <Popover.ArrowTip />
        </Popover.Arrow>
      ) : null}
      {children}
    </Popover.Content>
  );
};
