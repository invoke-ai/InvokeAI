import type { LayoutPreset, LayoutPresetId } from './layoutContracts';

import { getLayoutWidgetTypeIds } from './layoutWidgetSet';
import { loadWidgets, warmWidgets } from './widgetRegistry';

/**
 * Bound cold widget loading before applying the preset with per-widget fallbacks. Fully warm switches apply
 * synchronously and never consult the deadline.
 */
const APPLY_DEADLINE_MS = 250;

interface LayoutPresetActivatorDependencies {
  apply: (presetId: LayoutPresetId) => void;
  /** Test seam; omit for the production deadline. */
  applyDeadlineMs?: number;
  getActiveProjectId: () => string;
  isCurrent: (preset: LayoutPreset) => boolean;
  /** Whether every widget the preset renders is already in memory. */
  isLoaded: (preset: LayoutPreset) => boolean;
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
  isLoaded,
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

      // Apply warm presets synchronously in the click task; the already-advanced request id fences older cold
      // activations.
      if (isLoaded(preset)) {
        apply(preset.id);

        return preset.id;
      }

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
