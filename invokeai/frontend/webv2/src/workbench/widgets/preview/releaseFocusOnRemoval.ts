/**
 * The ref cleanup for a Preview control that a later render can remove while it holds focus (a filmstrip thumb, the
 * video, the header's ⋮): if the control leaves the document focused, focus goes to `getTarget()`, Preview's region by
 * default, so its keys keep working, unless something else has taken focus since. React also runs the cleanup when the
 * ref callback changes on a node that stays (the selection flipping on the click that focused it), so only a node really
 * gone gives focus away. Checked a microtask on, once the commit that runs the cleanup has removed the node.
 */
export const releaseFocusOnRemoval = (
  node: HTMLElement,
  getTarget: () => HTMLElement | null = () => node.closest<HTMLElement>('[role="region"]')
): (() => void) => {
  return () => {
    if (document.activeElement !== node) {
      return;
    }

    // Resolved while the node is still in place: a region found through it cannot be found once it is gone.
    const target = getTarget();

    queueMicrotask(() => {
      const { activeElement } = document;

      if (!node.isConnected && (activeElement === null || activeElement === document.body)) {
        target?.focus({ preventScroll: true });
      }
    });
  };
};
