import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import type { Atom } from 'nanostores';
import { computed, deepMap } from 'nanostores';
import type { RefObject } from 'react';
import { useEffect } from 'react';

const log = logger('system');

const REGION_NAMES = ['gallery', 'layers', 'canvas', 'workflows', 'viewer', 'settings'] as const;

type FocusRegionName = (typeof REGION_NAMES)[number];
type FocusRegionData = { name: FocusRegionName; targets: Set<HTMLElement> };
type FocusRegionState = {
  focusedRegion: FocusRegionName | null;
  regions: Record<FocusRegionName, FocusRegionData>;
};

const initialData = REGION_NAMES.reduce(
  (state, region) => {
    state.regions[region] = { name: region, targets: new Set() };
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
    acc[`$${region}`] = computed($focusRegionState, (state) => state.focusedRegion === region);
    return acc;
  },
  {} as Record<`$${FocusRegionName}`, Atom<boolean>>
);

const setFocus = (region: FocusRegionName | null) => {
  $focusRegionState.setKey('focusedRegion', region);
  log.trace(`Focus changed: ${region}`);
};

type UseFocusRegionOptions = {
  focusOnMount?: boolean;
};

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

    const regionData = $focusRegionState.get().regions[region];

    const targets = new Set(regionData.targets);
    targets.add(element);
    $focusRegionState.setKey(`regions.${region}.targets`, targets);

    if (focusOnMount) {
      setFocus(region);
    }

    return () => {
      const regionData = $focusRegionState.get().regions[region];
      const targets = new Set(regionData.targets);
      targets.delete(element);
      $focusRegionState.setKey(`regions.${region}.targets`, targets);

      if (targets.size === 0 && $focusRegionState.get().focusedRegion === region) {
        setFocus(null);
      }
    };
  }, [options, ref, region]);
};

export const useIsRegionFocused = (region: FocusRegionName) => {
  return useStore(FOCUS_REGIONS[`$${region}`]);
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
