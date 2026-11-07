import { useMountEffect } from '@platform/react/useMountEffect';

/**
 * Marks the prompt-trigger autocomplete list, the scroll container for its options.
 * Scrolling that list is the suggestion moving, not the anchored field leaving the viewport.
 */
const PROMPT_TRIGGER_AUTOCOMPLETE_ATTR = 'data-prompt-trigger-autocomplete';
export const PROMPT_TRIGGER_AUTOCOMPLETE_SCROLL_PROPS = { [PROMPT_TRIGGER_AUTOCOMPLETE_ATTR]: '' } as const;

/** Popover content and the autocomplete list. Descendants count: the event target may be an option. */
const EXEMPT_SCROLL_SELECTOR = `[data-scope="popover"][data-part="content"], [${PROMPT_TRIGGER_AUTOCOMPLETE_ATTR}]`;

/** Dismiss stale page-space anchors on scroll or resize, excluding scrolls inside an anchored surface. */
export const useDismissOnViewportChange = (enabled: boolean, dismiss: () => void): void => {
  useMountEffect(() => {
    if (!enabled) {
      return;
    }

    const handleViewportChange = (event: Event) => {
      // Resize targets the window, so it still dismisses. Only element scrolls can be internal.
      if (event.target instanceof Element && event.target.closest(EXEMPT_SCROLL_SELECTOR)) {
        return;
      }

      dismiss();
    };

    window.addEventListener('scroll', handleViewportChange, { capture: true, passive: true });
    window.addEventListener('resize', handleViewportChange, { passive: true });

    return () => {
      window.removeEventListener('scroll', handleViewportChange, { capture: true });
      window.removeEventListener('resize', handleViewportChange);
    };
  });
};

/** Mount this only while the anchored surface exists. */
export const DismissOnViewportChange = ({ dismiss, enabled }: { dismiss: () => void; enabled: boolean }) => {
  useDismissOnViewportChange(enabled, dismiss);
  return null;
};
