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
    /**
     * Resolves to the preset id that was applied, or `null` if this activation
     * was dropped — superseded by a later request, overtaken by a project
     * switch, or aimed at a preset definition the account has since replaced.
     *
     * The outcome is reported rather than swallowed because callers paint the
     * selection optimistically: a dropped activation that stayed silent would
     * leave a control showing a preset the store never adopted, with nothing
     * left to correct it.
     */
    activate: async (preset: LayoutPreset): Promise<LayoutPresetId | null> => {
      const projectId = getActiveProjectId();
      const requestId = ++latestRequestId;
      await waitForLoadOrDeadline(load(preset), applyDeadlineMs);

      if (requestId !== latestRequestId || projectId !== getActiveProjectId() || !isCurrent(preset)) {
        return null;
      }

      apply(preset.id);

      return preset.id;
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
