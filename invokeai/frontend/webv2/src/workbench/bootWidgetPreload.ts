import type { WidgetTypeId } from '@workbench/widgetContracts';

import { defaultLayoutPreset } from './layoutPresets';
import { getLayoutWidgetTypeIds } from './layoutWidgetSet';
import { getWidgetHosts, warmWidgets } from './widgetRegistry';

/**
 * Remember widget types in a separate boot hint so chunk downloads overlap hydration. Unknown or disabled types
 * are ignored; stale or cross-account hints only preload unused chunks.
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
 * Preload registry hosts and the hinted panel types during hydration, falling back to the default layout. The
 * implementation resource caches loads.
 */
export const preloadBootWidgets = (): void => {
  for (const widget of getWidgetHosts()) {
    widget.host?.preload();
  }

  warmWidgets(readBootWidgetHint() ?? getLayoutWidgetTypeIds(defaultLayoutPreset.snapshot));
};
