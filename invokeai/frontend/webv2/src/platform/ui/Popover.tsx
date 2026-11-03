import type { ComponentProps } from 'react';

import { Popover, usePopoverContext } from '@chakra-ui/react';

import { useRegisterWidgetOverlay } from './widgetOverlays';

type PopoverContentProps = ComponentProps<typeof Popover.Content> & {
  /**
   * Every popover anchored to a trigger points at it, as menus and hover cards do. Only listboxes that sit flush under
   * their field opt out (ModelSelect's picker, as Select and Combobox do).
   */
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
