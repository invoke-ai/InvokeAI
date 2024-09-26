import { logger } from 'app/logging/logger';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { objectKeys } from 'common/util/objectKeys';
import type { Atom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { RefObject } from 'react';
import { useEffect } from 'react';

const log = logger('system');

const _FOCUS_REGIONS = ['galleryPanel', 'layersPanel', 'canvas', 'stagingArea', 'workflows', 'imageViewer'] as const;

type FocusRegion = (typeof _FOCUS_REGIONS)[number];

// type FocusRegionData2 = { name: FocusRegion; focused: boolean; mounted: boolean; targets: Set<HTMLElement> };
// const initialData = _FOCUS_REGIONS.reduce(
//   (regionData, region) => {
//     regionData[region] = { name: region, focused: false, mounted: false, targets: new Set() };
//     return regionData;
//   },
//   {} as Record<FocusRegion, FocusRegionData2>
// );

// export const $regions = deepMap<Record<FocusRegion, FocusRegionData2>>(initialData);

export const $focusedRegion = atom<FocusRegion | null>(null);
type FocusRegionData = {
  targets: Set<HTMLElement>;
  $isFocused: Atom<boolean>;
};

export const FOCUS_REGIONS: Record<FocusRegion, FocusRegionData> = _FOCUS_REGIONS.reduce(
  (acc, region) => {
    acc[region] = {
      targets: new Set(),
      $isFocused: computed($focusedRegion, (focusedRegion) => focusedRegion === region),
    };
    return acc;
  },
  {} as Record<FocusRegion, FocusRegionData>
);

export const setFocus = (region: FocusRegion | null) => {
  $focusedRegion.set(region);
  log.trace(`Focus changed: ${$focusedRegion.get()}`);
};

export const useFocusRegion = (region: FocusRegion, ref: RefObject<HTMLElement>) => {
  useEffect(() => {
    if (!ref.current) {
      return;
    }

    const element = ref.current;
    FOCUS_REGIONS[region].targets.add(element);

    return () => {
      FOCUS_REGIONS[region].targets.delete(element);
    };
  }, [ref, region]);
};

type UseFocusRegionOnMountOptions = {
  mount?: boolean;
  unmount?: boolean;
};

export const useFocusRegionOnMount = (region: FocusRegion, options?: UseFocusRegionOnMountOptions) => {
  useEffect(() => {
    const { mount, unmount } = { mount: true, unmount: true, ...options };

    if (mount) {
      setFocus(region);
    }

    return () => {
      if (unmount && $focusedRegion.get() === region) {
        setFocus(null);
      }
    };
  }, [options, region]);
};

const onFocus = (_: FocusEvent) => {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLElement)) {
    return;
  }

  const regionCandidates: { region: FocusRegion; element: HTMLElement }[] = [];

  for (const region of objectKeys(FOCUS_REGIONS)) {
    for (const element of FOCUS_REGIONS[region].targets) {
      if (element.contains(activeElement)) {
        regionCandidates.push({ region, element });
      }
    }
  }

  let focusedRegion: FocusRegion | null = null;

  if (regionCandidates.length !== 0) {
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
    focusedRegion = regionCandidates[0]?.region ?? null;
  }

  setFocus(focusedRegion);
};

export const useFocusRegionWatcher = () => {
  useAssertSingleton('useFocusRegionWatcher');

  useEffect(() => {
    window.addEventListener('focus', onFocus, { capture: true });
    return () => {
      window.removeEventListener('focus', onFocus, { capture: true });
    };
  }, []);
};
