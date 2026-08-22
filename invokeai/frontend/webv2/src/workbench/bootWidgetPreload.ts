import type { WidgetTypeId } from '@workbench/widgetContracts';

import { defaultLayoutPreset } from './layoutPresets';
import { getLayoutWidgetTypeIds } from './layoutWidgetSet';
import { getWidgetHosts, warmWidgets } from './widgetRegistry';

/**
 * The widget implementations a boot will render are only knowable for certain
 * after the project has hydrated from the backend — but that hydration fetch
 * is exactly the window the network sits idle. This hint records the active
 * layout's widget set so the next boot can start those chunk downloads in
 * parallel with hydration instead of after it.
 *
 * Dedicated hint key rather than the workbench snapshot for the same reason as
 * the theme hint: the snapshot is per-user on multi-user backends, and a
 * preload is harmless when the signed-in user turns out to differ — unknown or
 * disabled type ids are ignored, and a stale set merely warms chunks the boot
 * would have fetched moments later.
 */
const BOOT_WIDGET_HINT_STORAGE_KEY = 'invokeai:v7:webv2:boot-widgets';

/** Sanity cap so a corrupt hint cannot fan out unbounded chunk fetches. */
const BOOT_WIDGET_HINT_LIMIT = 32;

export const readBootWidgetHint = (): WidgetTypeId[] | null => {
  try {
    const raw = window.localStorage.getItem(BOOT_WIDGET_HINT_STORAGE_KEY);

    if (!raw) {
      return null;
    }

    const parsed: unknown = JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      return null;
    }

    const typeIds = parsed.filter((value): value is WidgetTypeId => typeof value === 'string' && value.length > 0);

    return typeIds.length > 0 ? typeIds.slice(0, BOOT_WIDGET_HINT_LIMIT) : null;
  } catch {
    return null;
  }
};

export const writeBootWidgetHint = (typeIds: readonly WidgetTypeId[]): void => {
  try {
    window.localStorage.setItem(BOOT_WIDGET_HINT_STORAGE_KEY, JSON.stringify(typeIds.slice(0, BOOT_WIDGET_HINT_LIMIT)));
  } catch {
    // Storage unavailable — the next boot preloads the default layout's set.
  }
};

/**
 * Starts the widget chunk downloads a boot will need, called when the editor
 * route mounts — while project hydration is still in flight. Host widgets are
 * known statically from the registry; panel widgets come from the last boot's
 * hint, falling back to the default layout for a first run. Loads are cached
 * by the implementation resource, so a wrong guess costs one idle download and
 * a right one lets the shell mount without ever suspending.
 */
export const preloadBootWidgets = (): void => {
  for (const widget of getWidgetHosts()) {
    widget.host?.preload();
  }

  warmWidgets(readBootWidgetHint() ?? getLayoutWidgetTypeIds(defaultLayoutPreset.snapshot));
};
