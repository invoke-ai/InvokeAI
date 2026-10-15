const MODIFIER_ORDER = ['mod', 'ctrl', 'meta', 'shift', 'alt'] as const;

const KEY_ALIASES: Record<string, string> = {
  ' ': 'space',
  arrowdown: 'arrowdown',
  arrowleft: 'arrowleft',
  arrowright: 'arrowright',
  arrowup: 'arrowup',
  del: 'delete',
  escape: 'esc',
  return: 'enter',
};

const TINY_KEY_ALIASES: Record<string, string> = {
  '[': 'BracketLeft',
  ']': 'BracketRight',
  '.': 'Period',
  ',': 'Comma',
  '/': 'Slash',
  '\\': 'Backslash',
  '-': 'Minus',
  '=': 'Equal',
  arrowdown: 'ArrowDown',
  arrowleft: 'ArrowLeft',
  arrowright: 'ArrowRight',
  arrowup: 'ArrowUp',
  backspace: 'Backspace',
  delete: 'Delete',
  enter: 'Enter',
  esc: 'Escape',
  space: 'Space',
};

const TINY_MODIFIER_ALIASES: Record<string, string> = {
  alt: 'Alt',
  ctrl: 'Control',
  meta: 'Meta',
  mod: '$mod',
  shift: 'Shift',
};

export const IS_MAC_OS =
  typeof navigator !== 'undefined' &&
  (
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ??
    navigator.platform ??
    ''
  )
    .toLowerCase()
    .includes('mac');

export const normalizeHotkeyString = (hotkey: string): string => {
  const parts = hotkey
    .split('+')
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean)
    .map((part) => KEY_ALIASES[part] ?? part);
  const modifiers = parts
    .filter((part) => MODIFIER_ORDER.includes(part as (typeof MODIFIER_ORDER)[number]))
    .sort((left, right) => MODIFIER_ORDER.indexOf(left as never) - MODIFIER_ORDER.indexOf(right as never));
  const key = parts.find((part) => !MODIFIER_ORDER.includes(part as (typeof MODIFIER_ORDER)[number]));

  return key ? [...modifiers, key].join('+') : '';
};

/** `mod` is the platform's primary modifier (Cmd on macOS, Ctrl elsewhere); the other one, when held too, is kept. */
export const eventToHotkeyString = (event: KeyboardEvent, isMacOs = IS_MAC_OS): string => {
  if (event.isComposing || event.keyCode === 229 || ['Alt', 'Control', 'Meta', 'Shift'].includes(event.key)) {
    return '';
  }

  const modifiers: string[] = [];

  if (isMacOs ? event.metaKey : event.ctrlKey) {
    modifiers.push('mod');
  }
  if (isMacOs ? event.ctrlKey : event.metaKey) {
    modifiers.push(isMacOs ? 'ctrl' : 'meta');
  }
  if (event.shiftKey) {
    modifiers.push('shift');
  }
  if (event.altKey) {
    modifiers.push('alt');
  }

  return normalizeHotkeyString([...modifiers, event.key].join('+'));
};

export const toTinykeysBinding = (hotkey: string): string => {
  const normalized = normalizeHotkeyString(hotkey);
  const parts = normalized.split('+').filter(Boolean);

  return parts.map((part) => TINY_MODIFIER_ALIASES[part] ?? TINY_KEY_ALIASES[part] ?? part).join('+');
};

/** The `mod` key's platform name; a bare modifier normalizes to nothing, so use this rather than formatting 'mod'. */
export const MOD_KEY_LABEL = IS_MAC_OS ? 'cmd' : 'ctrl';

const formatPart = (part: string, isMacOs: boolean): string => {
  const named = part
    .trim()
    .toLowerCase()
    .replace('mod', isMacOs ? 'cmd' : 'ctrl');
  return isMacOs ? named.replace('alt', 'option') : named;
};

/** Fixed pointer gestures may use a bare modifier instead of a complete hotkey chord. */
export const formatHotkeyPartForPlatform = (part: string): string => formatPart(part, IS_MAC_OS);

/** Apple's menu order, Control Option Shift Command; normalization keeps its own order for comparison. */
const MAC_DISPLAY_MODIFIER_RANK: Record<string, number> = { cmd: 3, ctrl: 0, meta: 3, option: 1, shift: 2 };

export const formatHotkeyForPlatform = (hotkey: string, isMacOs: boolean = IS_MAC_OS): string[] => {
  const parts = normalizeHotkeyString(hotkey)
    .split('+')
    .filter(Boolean)
    .map((part) => formatPart(part, isMacOs));

  if (!isMacOs) {
    return parts;
  }

  const key = parts.at(-1)!;
  const modifiers = parts.slice(0, -1).sort((a, b) => MAC_DISPLAY_MODIFIER_RANK[a]! - MAC_DISPLAY_MODIFIER_RANK[b]!);
  return [...modifiers, key];
};

/** Text labels: adjacent modifier glyphs on macOS, word labels joined by + elsewhere. */
const MAC_KEY_LABELS: Record<string, string> = {
  alt: '⌥',
  cmd: '⌘',
  ctrl: '⌃',
  enter: '↵',
  option: '⌥',
  shift: '⇧',
};

const OTHER_KEY_LABELS: Record<string, string> = {
  alt: 'Alt',
  ctrl: 'Ctrl',
  enter: 'Enter',
  meta: 'Win',
  shift: 'Shift',
};

/** One formatted part's text label, for render sites that draw icons for some keys and need text for the rest. */
export const formatHotkeyPartLabel = (part: string, isMacOs: boolean = IS_MAC_OS): string =>
  (isMacOs ? MAC_KEY_LABELS[part] : OTHER_KEY_LABELS[part]) ?? part.toUpperCase();

/** A binding as one line of text, e.g. ⌘K on macOS and Ctrl+K elsewhere, for tooltips and inline copy. */
export const formatHotkeyLabel = (hotkey: string, isMacOs: boolean = IS_MAC_OS): string =>
  formatHotkeyForPlatform(hotkey, isMacOs)
    .map((part) => formatHotkeyPartLabel(part, isMacOs))
    .join(isMacOs ? '' : '+');

const ARIA_KEY_LABELS: Record<string, string> = {
  alt: 'Alt',
  cmd: 'Meta',
  ctrl: 'Control',
  enter: 'Enter',
  option: 'Alt',
  shift: 'Shift',
};

/** A binding in `aria-keyshortcuts` syntax. */
export const formatHotkeyAriaLabel = (hotkey: string, isMacOs: boolean = IS_MAC_OS): string =>
  formatHotkeyForPlatform(hotkey, isMacOs)
    .map((part) => ARIA_KEY_LABELS[part] ?? (part.length === 1 ? part.toUpperCase() : part))
    .join('+');

/** Inputs that take no typed text: a switch or checkbox owning focus has no native undo/shortcut to protect. */
const NON_TEXT_INPUT_TYPES = new Set([
  'button',
  'checkbox',
  'color',
  'file',
  'image',
  'radio',
  'range',
  'reset',
  'submit',
]);

export const isEditableHotkeyTarget = (target: EventTarget | null): boolean => {
  if (typeof HTMLElement === 'undefined') {
    return false;
  }

  if (!(target instanceof HTMLElement)) {
    return false;
  }

  const editable = target.closest('input, textarea, select, [contenteditable="true"]');

  return (
    editable !== null &&
    !(editable.tagName === 'INPUT' && NON_TEXT_INPUT_TYPES.has((editable as HTMLInputElement).type))
  );
};
