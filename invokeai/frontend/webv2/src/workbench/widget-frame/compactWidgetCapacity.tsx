import type { ReactNode } from 'react';

import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { createContext, use, useSyncExternalStore } from 'react';

/** The strip owns its clusters; a responsive chip receives remaining space without inspecting sibling widgets. */
export const createCompactWidgetCapacity = () => {
  const store = createExternalStoreCore({ width: Number.POSITIVE_INFINITY });
  let spacer: HTMLElement | null = null;
  let chip: HTMLElement | null = null;
  const measure = () => {
    const strip = spacer?.parentElement;
    if (!strip || !chip) {
      return;
    }
    const clusters = [...strip.children].filter((element) => element.hasAttribute('data-status-bar-cluster'));
    const style = getComputedStyle(strip);
    const occupied = clusters.reduce((width, cluster) => width + cluster.getBoundingClientRect().width, 0);
    // The chip's own width cancels out of occupied, so changing visible hints cannot change its capacity.
    const width = Math.max(
      24,
      strip.clientWidth -
        parseFloat(style.paddingLeft) -
        parseFloat(style.paddingRight) -
        occupied +
        chip.getBoundingClientRect().width
    );
    if (width !== store.getSnapshot().width) {
      store.setSnapshot({ width });
    }
  };
  return {
    ...store,
    bindSpacer: (element: HTMLDivElement | null) => {
      spacer = element;
      if (!element) {
        return;
      }
      const observer = new ResizeObserver(measure);
      observer.observe(element);
      const strip = element.parentElement;
      if (strip) {
        observer.observe(strip);
        for (const cluster of strip.children) {
          if (cluster.hasAttribute('data-status-bar-cluster')) {
            observer.observe(cluster);
          }
        }
      }
      measure();
      return () => {
        observer.disconnect();
        spacer = null;
      };
    },
    bindChip: (element: HTMLDivElement | null) => {
      chip = element;
      measure();
      return () => {
        if (chip === element) {
          chip = null;
        }
      };
    },
  };
};

const CompactWidgetCapacityContext = createContext<ReturnType<typeof createCompactWidgetCapacity> | null>(null);
export const CompactWidgetCapacityProvider = ({
  children,
  capacity,
}: {
  children: ReactNode;
  capacity: ReturnType<typeof createCompactWidgetCapacity>;
}) => <CompactWidgetCapacityContext value={capacity}>{children}</CompactWidgetCapacityContext>;
export const useCompactWidgetCapacityResource = () => use(CompactWidgetCapacityContext);
const subscribeToNothing = () => () => {};
const UNBOUNDED_CAPACITY = { width: Number.POSITIVE_INFINITY };
const getUnboundedCapacity = () => UNBOUNDED_CAPACITY;
export const useCompactWidgetCapacity = (): number => {
  const capacity = useCompactWidgetCapacityResource();
  return useSyncExternalStore(capacity?.subscribe ?? subscribeToNothing, capacity?.getSnapshot ?? getUnboundedCapacity)
    .width;
};
