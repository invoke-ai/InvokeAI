import { useMountEffect } from '@platform/react/useMountEffect';

/** Dismiss stale page-space anchors on scrolling, excluding the popover's own scroll. */
export const useDismissOnViewportChange = (enabled: boolean, dismiss: () => void): void => {
  useMountEffect(() => {
    if (!enabled) {
      return;
    }

    const handleViewportChange = (event: Event) => {
      if (event.target instanceof Element && event.target.closest('[data-scope="popover"][data-part="content"]')) {
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
