import type { ComponentProps, ReactNode } from 'react';

import { MenuContent } from './Menu';
import { Scrollable } from './Scrollable';

/** A long menu's height: the context menus' cap, shrunk to the room the positioner measured beside its anchor. */
const SCROLLABLE_MENU_MAX_H = 'min(28rem, var(--available-height))';

/**
 * Re-implements Zag's highlight-follow. Zag scrolls a highlighted item into view only when the menu content itself
 * scrolls; here the themed Scrollable viewport inside it does, because the app keeps its overlay scrollbar over native
 * ones in menus too. The cost is this observer: keyboard and typeahead highlights are followed here, and a pointer
 * highlight is skipped (the item under the pointer is already in view), as Zag does, so hovering an edge row does not
 * nudge the list.
 */
const followHighlightedItem = (viewport: HTMLDivElement | null) => {
  if (!viewport) {
    return;
  }
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      const item = record.target;
      if (item instanceof HTMLElement && item.hasAttribute('data-highlighted') && !item.matches(':hover')) {
        item.scrollIntoView({ block: 'nearest' });
      }
    }
  });
  observer.observe(viewport, { attributeFilter: ['data-highlighted'], subtree: true });
  return () => observer.disconnect();
};

/**
 * Menu content for long lists, such as boards to move items to: it grows to the shared cap or the available height,
 * then scrolls in a Scrollable. The column layout gives the Scrollable the content's padded box, so the last row is
 * never cut off by the content's padding.
 */
export const ScrollableMenuContent = ({
  children,
  ...props
}: ComponentProps<typeof MenuContent> & { children: ReactNode }) => (
  <MenuContent display="flex" flexDirection="column" maxH={SCROLLABLE_MENU_MAX_H} overflow="hidden" {...props}>
    <Scrollable minH="0" viewportRef={followHighlightedItem}>
      {children}
    </Scrollable>
  </MenuContent>
);
