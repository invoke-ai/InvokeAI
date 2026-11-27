import type { ComponentProps, ReactNode } from 'react';

import { MenuContent, SCROLLABLE_MENU_CONTENT_PROPS } from './Menu';
import { Scrollable } from './Scrollable';

/**
 * Re-implements Zag's highlight-follow. Zag scrolls a highlighted item into view only when the menu content itself
 * scrolls; here the themed Scrollable viewport inside it does, because the app keeps its overlay scrollbar over native
 * ones in menus too. The cost is this observer: keyboard and typeahead highlights are followed here, and a pointer
 * highlight is skipped (the item under the pointer is already in view), as Zag does, so hovering an edge row does not
 * nudge the list.
 */
const followHighlightedItem = (viewport: HTMLDivElement) => {
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
 * Keeps the viewport out of the tab order: the menu owns keyboard navigation, and a menu may not contain a focusable
 * element, even at tabindex -1. Zag sets tabindex 0 until it first measures its scrollbars (and whenever both axes
 * overflow), and an undefined prop cannot override it, so the attribute is removed on attach, before first paint, and
 * whenever Zag sets it again.
 */
const keepOutOfTabOrder = (viewport: HTMLDivElement) => {
  viewport.removeAttribute('tabindex');
  const observer = new MutationObserver(() => viewport.removeAttribute('tabindex'));
  observer.observe(viewport, { attributeFilter: ['tabindex'] });
  return () => observer.disconnect();
};

const attachMenuViewport = (viewport: HTMLDivElement | null) => {
  if (!viewport) {
    return;
  }
  const stopKeepingOutOfTabOrder = keepOutOfTabOrder(viewport);
  const stopFollowingHighlight = followHighlightedItem(viewport);
  return () => {
    stopKeepingOutOfTabOrder();
    stopFollowingHighlight();
  };
};

/** A menu's scrolling list, in content laid out with `SCROLLABLE_MENU_CONTENT_PROPS`, above any rows that stay in view. */
export const ScrollableMenuList = ({ children }: { children: ReactNode }) => (
  <Scrollable minH="0" viewportRef={attachMenuViewport}>
    {children}
  </Scrollable>
);

/** Menu content for long lists, such as boards to move items to: the list scrolls within the menu's cap. */
export const ScrollableMenuContent = ({
  children,
  ...props
}: ComponentProps<typeof MenuContent> & { children: ReactNode }) => (
  <MenuContent {...SCROLLABLE_MENU_CONTENT_PROPS} {...props}>
    <ScrollableMenuList>{children}</ScrollableMenuList>
  </MenuContent>
);
