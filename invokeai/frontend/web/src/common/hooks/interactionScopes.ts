import { logger } from 'app/logging/logger';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import type { Atom } from 'nanostores';
import { computed, deepMap } from 'nanostores';
import type { RefObject } from 'react';
import { useEffect } from 'react';

const log = logger('system');

const REGION_NAMES = ['galleryPanel', 'layersPanel', 'canvas', 'workflows', 'imageViewer'] as const;

type FocusRegionName = (typeof REGION_NAMES)[number];
type FocusRegionData = { name: FocusRegionName; mounted: number; targets: Set<HTMLElement> };
type FocusRegionState = {
  focusedRegion: FocusRegionName | null;
  regions: Record<FocusRegionName, FocusRegionData>;
};

const initialData = REGION_NAMES.reduce(
  (state, region) => {
    state.regions[region] = { name: region, mounted: 0, targets: new Set() };
    return state;
  },
  {
    focusedRegion: null,
    regions: {},
  } as FocusRegionState
);

const $focusRegionState = deepMap<FocusRegionState>(initialData);
export const $focusedRegion = computed($focusRegionState, (regions) => regions.focusedRegion);
export const FOCUS_REGIONS = REGION_NAMES.reduce(
  (acc, region) => {
    acc[`$${region}`] = computed($focusRegionState, (state) => ({
      isFocused: state.focusedRegion === region,
      isMounted: state.regions[region].mounted > 0,
    }));
    return acc;
  },
  {} as Record<`$${FocusRegionName}`, Atom<{ isFocused: boolean; isMounted: boolean }>>
);

export const setFocus = (region: FocusRegionName | null) => {
  $focusRegionState.setKey('focusedRegion', region);
  log.trace(`Focus changed: ${region}`);
};

export const useFocusRegion = (region: FocusRegionName, ref: RefObject<HTMLElement>) => {
  useEffect(() => {
    if (!ref.current) {
      return;
    }

    const element = ref.current;

    const regionData = $focusRegionState.get().regions[region];

    const targets = new Set(regionData.targets);
    targets.add(element);
    $focusRegionState.setKey(`regions.${region}.targets`, targets);

    return () => {
      const regionData = $focusRegionState.get().regions[region];
      const targets = new Set(regionData.targets);
      targets.delete(element);
      $focusRegionState.setKey(`regions.${region}.targets`, targets);
    };
  }, [ref, region]);
};

export const useFocusRegionOnMount = (region: FocusRegionName) => {
  useEffect(() => {
    const mounted = $focusRegionState.get().regions[region].mounted + 1;
    $focusRegionState.setKey(`regions.${region}.mounted`, mounted);
    setFocus(region);
    log.trace(`Focus region ${region} mounted, count: ${mounted}`);

    return () => {
      let mounted = $focusRegionState.get().regions[region].mounted - 1;
      if (mounted < 0) {
        log.warn(`Focus region ${region} mounted count is negative: ${mounted}!`);
        mounted = 0;
      } else {
        log.trace(`Focus region ${region} unmounted, count: ${mounted}`);
      }
      $focusRegionState.setKey(`regions.${region}.mounted`, mounted);
      if (mounted === 0) {
        setFocus(null);
      }
    };
  }, [region]);
};

const onFocus = (_: FocusEvent) => {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLElement)) {
    return;
  }

  const regionCandidates: { region: FocusRegionName; element: HTMLElement }[] = [];

  const state = $focusRegionState.get();
  for (const regionData of Object.values(state.regions)) {
    for (const element of regionData.targets) {
      if (element.contains(activeElement)) {
        regionCandidates.push({ region: regionData.name, element });
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

export const useFocusRegionWatcher = () => {
  useAssertSingleton('useFocusRegionWatcher');

  useEffect(() => {
    window.addEventListener('focus', onFocus, { capture: true });
    return () => {
      window.removeEventListener('focus', onFocus, { capture: true });
    };
  }, []);
};
