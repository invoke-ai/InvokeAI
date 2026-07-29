import { useMountEffect } from '@platform/react/useMountEffect';

/**
 * Dismisses a rect-anchored surface once scrolling or resizing makes its stored
 * viewport coordinates stale.
 */
export const useDismissOnViewportChange = (enabled: boolean, dismiss: () => void): void => {
  useMountEffect(() => {
    if (!enabled) {
      return;
    }

    const handleViewportChange = () => dismiss();

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
