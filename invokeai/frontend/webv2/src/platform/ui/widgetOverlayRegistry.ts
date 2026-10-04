/** React-free dismissal epochs and pending closes; Workbench bumps them before hiding widgets. */

let epoch = 0;
const subscribers = new Set<() => void>();
const closers = new Set<() => void>();

export const subscribeToWidgetOverlayEpoch = (listener: () => void): (() => void) => {
  subscribers.add(listener);
  return () => {
    subscribers.delete(listener);
  };
};

export const getWidgetOverlayEpoch = (): number => epoch;

export const registerWidgetOverlayCloser = (close: () => void): (() => void) => {
  closers.add(close);
  return () => {
    closers.delete(close);
  };
};

/**
 * Call before hiding widgets: deferred library closes may be dropped. Epochs hide content synchronously; all
 * widget overlays close, while shell overlays remain.
 */
export const closeWidgetOverlays = (): void => {
  epoch += 1;
  for (const notify of Array.from(subscribers)) {
    notify();
  }
  for (const close of Array.from(closers)) {
    close();
  }
};
