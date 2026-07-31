/**
 * Shared state behind every {@link ColorPicker}: the default swatch palette, a
 * most-recently-used list of committed colors, and the preferred channel
 * format.
 *
 * Both persisted values are per browser rather than per account or project —
 * like the command palette's frecency record, a picker's muscle memory should
 * not reset when the user switches projects. Storage failures (quota, private
 * mode, blocked cookies) are non-fatal: the picker still works, it just forgets.
 */

import { createExternalStore } from '@platform/state/externalStore';

/** The channel layouts the picker can present. */
export type ColorPickerFormat = 'hex' | 'rgb' | 'hsl' | 'hsb';

export const COLOR_PICKER_FORMATS: readonly ColorPickerFormat[] = ['hex', 'rgb', 'hsl', 'hsb'];

const RECENTS_STORAGE_KEY = 'invokeai:v7:webv2:color-recents';
const FORMAT_STORAGE_KEY = 'invokeai:v7:webv2:color-format';
const MAX_RECENTS = 12;

/**
 * The workbench palette: neutrals plus the regional-guidance mask hues, so a
 * brush color and a mask fill picked from defaults read as one system.
 *
 * These values intentionally duplicate `REGIONAL_GUIDANCE_FILL_COLORS` in the
 * canvas engine. `platform/` may not import `workbench/` (`platform-independence`)
 * and the engine may not import back out (`canvas-engine-independence`), so a
 * shared constant would have to live in a third module owned by neither.
 */
export const DEFAULT_COLOR_SWATCHES: readonly string[] = [
  '#000000',
  '#ffffff',
  '#808080',
  '#e07575',
  '#dc9065',
  '#fae150',
  '#83d683',
  '#799ddb',
  '#a178d6',
  '#d58bca',
];

const isColorPickerFormat = (value: unknown): value is ColorPickerFormat =>
  typeof value === 'string' && (COLOR_PICKER_FORMATS as readonly string[]).includes(value);

/**
 * Reads a persisted string. Touching `localStorage` at all can throw
 * (SecurityError when storage is disabled), not just `getItem` — both stay
 * inside the guard.
 */
const readStored = (key: string): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
};

const writeStored = (key: string, value: string): void => {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Quota or private-mode failures are non-fatal; these are conveniences.
  }
};

const parseRecents = (raw: string | null): string[] => {
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter((item): item is string => typeof item === 'string').slice(0, MAX_RECENTS);
  } catch {
    return [];
  }
};

interface ColorPickerSnapshot {
  format: ColorPickerFormat;
  recents: string[];
}

const readStoredFormat = (): ColorPickerFormat => {
  const stored = readStored(FORMAT_STORAGE_KEY);

  return isColorPickerFormat(stored) ? stored : 'hex';
};

const store = createExternalStore<ColorPickerSnapshot>({
  format: readStoredFormat(),
  recents: parseRecents(readStored(RECENTS_STORAGE_KEY)),
});

/**
 * Promotes `color` to the front of the recents list. Callers record on commit
 * (`onValueChangeEnd`) only — recording every drag frame would fill the list
 * with the colors swept through on the way to the chosen one.
 */
export const recordRecentColor = (color: string): void => {
  const normalized = color.toLowerCase();
  const { recents } = store.getSnapshot();

  if (recents[0] === normalized) {
    return;
  }

  const next = [normalized, ...recents.filter((entry) => entry !== normalized)].slice(0, MAX_RECENTS);

  store.patchSnapshot({ recents: next });
  writeStored(RECENTS_STORAGE_KEY, JSON.stringify(next));
};

export const setColorPickerFormat = (format: ColorPickerFormat): void => {
  store.patchSnapshot({ format });
  writeStored(FORMAT_STORAGE_KEY, format);
};

export const useRecentColors = (): string[] => store.useSelector((snapshot) => snapshot.recents);

export const useColorPickerFormat = (): ColorPickerFormat =>
  store.useSelector((snapshot) => snapshot.format, Object.is);
