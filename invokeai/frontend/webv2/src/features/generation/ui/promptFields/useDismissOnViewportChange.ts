import { useMountEffect } from '@platform/react/useMountEffect';

/**
 * Anchored surfaces whose own scrolling is not a viewport change: Chakra popover content (the trigger
 * picker) and the prompt autocomplete list, which is a plain listbox without popover markup.
 */
const OWN_SCROLL_SELECTOR = '[data-scope="popover"][data-part="content"], [data-prompt-trigger-autocomplete]';

/** Dismiss stale page-space anchors on scrolling, excluding the anchored surface's own scroll. */
export const useDismissOnViewportChange = (enabled: boolean, dismiss: () => void): void => {
  useMountEffect(() => {
    if (!enabled) {
      return;
    }

    const handleViewportChange = (event: Event) => {
      if (event.target instanceof Element && event.target.closest(OWN_SCROLL_SELECTOR)) {
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
