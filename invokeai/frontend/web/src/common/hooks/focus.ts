import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { RefObject } from 'react';
import { useEffect } from 'react';
import { objectKeys } from 'tsafe';

/**
 * We need to manage focus regions to conditionally enable hotkeys:
 * - Some hotkeys should only be enabled when a specific region is focused.
 * - Some hotkeys may conflict with other regions, so we need to disable them when a specific region is focused. For
 *  example, `esc` is used to clear the gallery selection, but it is also used to cancel a filter or transform on the
 *  canvas.
 *
 * To manage focus regions, we use a system of hooks and stores:
 * - `useFocusRegion` is a hook that registers an element as part of a focus region. When that element is focused, by
 *  click or any other action, that region is set as the focused region. Optionally, focus can be set on mount. This
 *  is useful for components like the image viewer.
 * - `useIsRegionFocused` is a hook that returns a boolean indicating if a specific region is focused.
 * - `useFocusRegionWatcher` is a hook that listens for focus events on the window. When an element is focused, it
 *  checks if it is part of a focus region and sets that region as the focused region.
 */

//

const log = logger('system');

/**
 * The names of the focus regions.
 */
type FocusRegionName = 'gallery' | 'layers' | 'canvas' | 'workflows' | 'viewer';

/**
 * A map of focus regions to the elements that are part of that region.
 */
const REGION_TARGETS: Record<FocusRegionName, Set<HTMLElement>> = {
  gallery: new Set<HTMLElement>(),
  layers: new Set<HTMLElement>(),
  canvas: new Set<HTMLElement>(),
  workflows: new Set<HTMLElement>(),
  viewer: new Set<HTMLElement>(),
} as const;

/**
 * The currently-focused region or `null` if no region is focused.
 */
const $focusedRegion = atom<FocusRegionName | null>(null);

/**
 * A map of focus regions to atoms that indicate if that region is focused.
 */
const FOCUS_REGIONS = objectKeys(REGION_TARGETS).reduce(
  (acc, region) => {
    acc[`$${region}`] = computed($focusedRegion, (focusedRegion) => focusedRegion === region);
    return acc;
  },
  {} as Record<`$${FocusRegionName}`, Atom<boolean>>
);

/**
 * Sets the focused region, logging a trace level message.
 */
const setFocus = (region: FocusRegionName | null) => {
  $focusedRegion.set(region);
  log.trace(`Focus changed: ${region}`);
};

type UseFocusRegionOptions = {
  focusOnMount?: boolean;
};

/**
 * Registers an element as part of a focus region. When that element is focused, by click or any other action, that
 * region is set as the focused region. Optionally, focus can be set on mount.
 *
 * On unmount, if the element is the last element in the region and the region is focused, the focused region is set to
 * `null`.
 *
 * @param region The focus region name.
 * @param ref The ref of the element to register.
 * @param options The options.
 */
export const useFocusRegion = (
  region: FocusRegionName,
  ref: RefObject<HTMLElement>,
  options?: UseFocusRegionOptions
) => {
  useEffect(() => {
    if (!ref.current) {
      return;
    }

    const { focusOnMount = false } = { focusOnMount: false, ...options };

    const element = ref.current;

    REGION_TARGETS[region].add(element);

    if (focusOnMount) {
      setFocus(region);
    }

    return () => {
      REGION_TARGETS[region].delete(element);

      if (REGION_TARGETS[region].size === 0 && $focusedRegion.get() === region) {
        setFocus(null);
      }
    };
  }, [options, ref, region]);
};

/**
 * Returns a boolean indicating if a specific region is focused.
 * @param region The focus region name.
 */
export const useIsRegionFocused = (region: FocusRegionName) => {
  return useStore(FOCUS_REGIONS[`$${region}`]);
};

/**
 * Listens for focus events on the window. When an element is focused, it checks if it is part of a focus region and sets
 * that region as the focused region. The region corresponding to the deepest element is set.
 */
const onFocus = (_: FocusEvent) => {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLElement)) {
    return;
  }

  const regionCandidates: { region: FocusRegionName; element: HTMLElement }[] = [];

  for (const region of objectKeys(REGION_TARGETS)) {
    for (const element of REGION_TARGETS[region]) {
      if (element.contains(activeElement)) {
        regionCandidates.push({ region, element });
      }
    }
  }

  if (regionCandidates.length === 0) {
    return;
  }

  // Sort by the shallowest element
  regionCandidates.sort((a, b) => {
    if (b.element.contains(a.element)) {
      return -1;
    }
    if (a.element.contains(b.element)) {
      return 1;
    }
    return 0;
  });

  // Set the region of the deepest element
  const focusedRegion = regionCandidates[0]?.region;

  if (!focusedRegion) {
    log.warn('No focused region found');
    return;
  }

  setFocus(focusedRegion);
};

/**
 * Listens for focus events on the window. When an element is focused, it checks if it is part of a focus region and sets
 * that region as the focused region. This is a singleton.
 */
export const useFocusRegionWatcher = () => {
  useAssertSingleton('useFocusRegionWatcher');

  useEffect(() => {
    window.addEventListener('focus', onFocus, { capture: true });
    return () => {
      window.removeEventListener('focus', onFocus, { capture: true });
    };
  }, []);
};
