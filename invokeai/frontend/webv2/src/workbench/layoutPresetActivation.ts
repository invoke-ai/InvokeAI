import type { LayoutPreset, LayoutPresetId } from './layoutContracts';

import { getLayoutWidgetTypeIds } from './layoutWidgetSet';
import { loadWidgets, warmWidgets } from './widgetRegistry';

/**
 * How long an activation waits for widget implementations before applying the
 * preset anyway. Warm widgets — the common case, given boot preloading and the
 * strip's hover preload — settle immediately and keep the atomic, no-fallback
 * reveal. A cold or failing chunk stops gating the entire layout at this
 * deadline: the switch commits and the stragglers reveal progressively behind
 * their own per-widget fallbacks.
 */
const APPLY_DEADLINE_MS = 250;

interface LayoutPresetActivatorDependencies {
  apply: (presetId: LayoutPresetId) => void;
  /** Test seam; omit for the production deadline. */
  applyDeadlineMs?: number;
  getActiveProjectId: () => string;
  isCurrent: (preset: LayoutPreset) => boolean;
  load: (preset: LayoutPreset) => Promise<unknown>;
}

const waitForLoadOrDeadline = async (pending: Promise<unknown>, deadlineMs: number): Promise<void> => {
  let timer: ReturnType<typeof setTimeout> | undefined;

  try {
    await Promise.race([
      pending,
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, deadlineMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};

export const createLayoutPresetActivator = ({
  apply,
  applyDeadlineMs = APPLY_DEADLINE_MS,
  getActiveProjectId,
  isCurrent,
  load,
}: LayoutPresetActivatorDependencies) => {
  let latestRequestId = 0;

  return {
    activate: async (preset: LayoutPreset): Promise<void> => {
      const projectId = getActiveProjectId();
      const requestId = ++latestRequestId;
      await waitForLoadOrDeadline(load(preset), applyDeadlineMs);

      if (requestId === latestRequestId && projectId === getActiveProjectId() && isCurrent(preset)) {
        apply(preset.id);
      }
    },
    invalidate: (): void => {
      latestRequestId += 1;
    },
  };
};

export const preloadLayoutPresetWidgets = (preset: LayoutPreset): void => {
  warmWidgets(getLayoutWidgetTypeIds(preset.snapshot));
};

export const loadLayoutPresetWidgets = (preset: LayoutPreset): Promise<unknown> =>
  loadWidgets(getLayoutWidgetTypeIds(preset.snapshot));
