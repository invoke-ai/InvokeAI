import { createExternalStore } from '@platform/state/externalStore';

export type ColorPickerFormat = 'hex' | 'rgb' | 'hsl' | 'hsb';

export const COLOR_PICKER_FORMATS: readonly ColorPickerFormat[] = ['hex', 'rgb', 'hsl', 'hsb'];

/** How the color pane's picking surface renders: the HSV wheel+triangle or the box with a hue strip. */
export type ColorPickerMode = 'wheel' | 'box';

const RECENTS_STORAGE_KEY = 'invokeai:v7:webv2:color-recents';
const FORMAT_STORAGE_KEY = 'invokeai:v7:webv2:color-format';
const MODE_STORAGE_KEY = 'invokeai:v7:webv2:color-picker-mode';
const MAX_RECENTS = 12;

/** Neutral swatches followed by the Canvas mask palette. */
export const DEFAULT_COLOR_SWATCHES: readonly string[] = [
  '#000000',
  '#ffffff',
  '#808080',
  '#e07575',
  '#83d683',
  '#F59E42',
  '#F4D35E',
  '#38C9C5',
  '#62B5F5',
  '#637BE8',
  '#AD80E8',
  '#ED79BA',
  '#B7D65A',
];

const isColorPickerFormat = (value: unknown): value is ColorPickerFormat =>
  typeof value === 'string' && (COLOR_PICKER_FORMATS as readonly string[]).includes(value);

const isColorPickerMode = (value: unknown): value is ColorPickerMode => value === 'wheel' || value === 'box';

/** Guard access to localStorage itself; the property getter can throw before getItem. */
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
  mode: ColorPickerMode;
  recents: string[];
}

const readStoredFormat = (): ColorPickerFormat => {
  const stored = readStored(FORMAT_STORAGE_KEY);

  return isColorPickerFormat(stored) ? stored : 'hex';
};

const readStoredMode = (): ColorPickerMode => {
  const stored = readStored(MODE_STORAGE_KEY);

  return isColorPickerMode(stored) ? stored : 'wheel';
};

const store = createExternalStore<ColorPickerSnapshot>({
  format: readStoredFormat(),
  mode: readStoredMode(),
  recents: parseRecents(readStored(RECENTS_STORAGE_KEY)),
});

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

export const setColorPickerMode = (mode: ColorPickerMode): void => {
  store.patchSnapshot({ mode });
  writeStored(MODE_STORAGE_KEY, mode);
};

export const useColorPickerMode = (): ColorPickerMode => store.useSelector((snapshot) => snapshot.mode, Object.is);

export const useRecentColors = (): string[] => store.useSelector((snapshot) => snapshot.recents);

export const useColorPickerFormat = (): ColorPickerFormat =>
  store.useSelector((snapshot) => snapshot.format, Object.is);
