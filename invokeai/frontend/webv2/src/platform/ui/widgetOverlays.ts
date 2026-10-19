import { useMountEffect } from '@platform/react/useMountEffect';
import { createContext, useContext, useState, useSyncExternalStore } from 'react';

import {
  getWidgetOverlayEpoch,
  registerWidgetOverlayCloser,
  subscribeToWidgetOverlayEpoch,
} from './widgetOverlayRegistry';

export { closeWidgetOverlays } from './widgetOverlayRegistry';

/** Associates overlays with the widget host for closeWidgetOverlays. */
export const WidgetOverlayOwnerContext = createContext(false);

const subscribeToNothing = (): (() => void) => () => undefined;

/**
 * Register while mounted; hide content synchronously after dismissal and resend the library close when a hidden
 * widget returns.
 */
export const useRegisterWidgetOverlay = (open: boolean, setOpen: (open: boolean) => void): boolean => {
  const ownedByWidget = useContext(WidgetOverlayOwnerContext);
  const current = useSyncExternalStore(
    ownedByWidget ? subscribeToWidgetOverlayEpoch : subscribeToNothing,
    getWidgetOverlayEpoch,
    getWidgetOverlayEpoch
  );
  // The epoch this opening belongs to, reset whenever the overlay reopens.
  const [opening, setOpening] = useState({ epoch: current, open });
  if (opening.open !== open) {
    setOpening({ epoch: current, open });
  }
  const stale = ownedByWidget && open && opening.epoch !== current;
  // Read the current dismissal epoch on remount so closes issued while hidden still apply.
  useMountEffect(() => {
    if (!ownedByWidget) {
      return;
    }
    if (open && opening.epoch !== getWidgetOverlayEpoch()) {
      setOpen(false);
    }
    return registerWidgetOverlayCloser(() => setOpen(false));
  });
  return stale;
};
