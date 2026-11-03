import { matchKeybindingPress, parseKeybinding } from 'tinykeys';

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

/** A Latin letter or digit, as a binding names a key. */
const isLatinKey = (key: string): boolean => /^[a-z0-9]$/i.test(key);

/** The physical key of a Latin letter or digit (`KeyN`, `Digit1`). */
const toPhysicalKeyCode = (key: string): string => (/^[0-9]$/.test(key) ? `Digit${key}` : `Key${key.toUpperCase()}`);

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

  // A key is recorded as typed, so the binding keeps its meaning on any layout. Only a chord the matching falls back
  // to position for (Alt with Cmd, Ctrl or Meta) records a dead key or symbol by its position: there macOS's Option
  // reports `Dead` or `˜` for N. Option alone keeps the typed character (`alt+å`), as it matches only by character.
  // AltGr typed a character of the layout (`ń`): that character is the binding, as matching refuses position under it.
  const fallsBackToPosition =
    event.altKey &&
    modifiers.some((modifier) => modifier !== 'shift' && modifier !== 'alt') &&
    event.getModifierState?.('AltGraph') !== true;
  const physical =
    fallsBackToPosition && !isLatinKey(event.key) ? /^(?:Key([A-Z])|Digit([0-9]))$/.exec(event.code ?? '') : null;
  const key = physical ? (physical[1] ?? physical[2]!).toLowerCase() : event.key;

  return normalizeHotkeyString([...modifiers, key].join('+'));
};

/**
 * Whether a binding may also match its letter or digit by position: an Alt chord with Cmd, Ctrl or Meta, which types
 * no character. macOS's Option still changes the key such a chord reports (`Dead`, `˜`), so `mod+alt+n` would never
 * fire by character. An Alt chord alone keeps to the character, so Option+A still types `å` in a text field.
 */
const hasPositionFallback = (parts: readonly string[]): boolean => {
  const key = parts.at(-1) ?? '';

  return (
    parts.includes('alt') &&
    (parts.includes('mod') || parts.includes('ctrl') || parts.includes('meta')) &&
    isLatinKey(key) &&
    key.length === 1
  );
};

export const toTinykeysBinding = (hotkey: string): string => {
  const normalized = normalizeHotkeyString(hotkey);
  const parts = normalized.split('+').filter(Boolean);
  const key = parts.at(-1) ?? '';
  // The character, or its position (tinykeys tests a `(…)` pattern against both the key and the code);
  // `acceptsHotkeyPress` admits the position only when the press typed no Latin character.
  const positional = hasPositionFallback(parts) ? `(${key}|${toPhysicalKeyCode(key)})` : null;

  return parts
    .map((part, index) =>
      index === parts.length - 1 && positional
        ? positional
        : (TINY_MODIFIER_ALIASES[part] ?? TINY_KEY_ALIASES[part] ?? part)
    )
    .join('+');
};

/**
 * The checks a press must pass beyond the tinykeys match, for a binding matched by its key's position: the press must
 * have typed no Latin character, so a non-QWERTY layout's letters keep their meaning; and it must not hold AltGr, which
 * some layouts need to type a character at that position (`ń`). Every other match stands as tinykeys made it,
 * including its reading of AltGr as Alt (browsers report AltGr for Ctrl+Alt on those layouts).
 */
export const acceptsHotkeyPress = (event: KeyboardEvent, hotkey: string): boolean => {
  const parts = normalizeHotkeyString(hotkey).split('+').filter(Boolean);
  const key = parts.at(-1) ?? '';

  if (!hasPositionFallback(parts) || event.key.toLowerCase() === key) {
    return true;
  }

  return (
    !isLatinKey(event.key) && event.code === toPhysicalKeyCode(key) && event.getModifierState?.('AltGraph') !== true
  );
};

/**
 * Whether a press is this binding, matched exactly as the hotkey runtime's tinykeys listener matches it, so a local
 * handler and the window runtime agree on every press.
 */
export const isHotkeyPress = (event: KeyboardEvent, hotkey: string): boolean => {
  const binding = toTinykeysBinding(hotkey);
  const [press] = binding ? parseKeybinding(binding) : [];
  return press !== undefined && matchKeybindingPress(event, press) && acceptsHotkeyPress(event, hotkey);
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

/** What a focused radio group or slider does with a key itself: move its selection or value. */
const NATIVE_CONTROL_KEYS = new Set(['ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'End', 'Home']);
const ACTIVATION_KEYS = new Set([' ', 'Enter']);
/** Controls a bare Space or Enter activates natively: buttons, links, toggles, tabs, menu items, options. */
const ACTIVATABLE_CONTROL_SELECTOR = [
  'a[href]',
  'button',
  'input[type="button"]',
  'input[type="checkbox"]',
  'input[type="radio"]',
  'input[type="reset"]',
  'input[type="submit"]',
  'summary',
  '[role="button"]',
  '[role="checkbox"]',
  '[role="link"]',
  '[role="menuitem"]',
  '[role="menuitemcheckbox"]',
  '[role="menuitemradio"]',
  '[role="option"]',
  '[role="switch"]',
  '[role="tab"]',
].join(', ');

/**
 * Whether a press is the focused control's own: a radio group's or slider's navigation (arrows, Home, End), or the
 * Space or Enter that activates a focused button, link, toggle or tab, all without Ctrl, Meta or Alt. These controls
 * take no typed text, so {@link isEditableHotkeyTarget} lets chords through from them, but a plain shortcut on the
 * same key (Gallery's arrow navigation, the workflow editor's Space for Add node) would prevent the default and freeze
 * or skip the control.
 */
export const isNativeControlKeyPress = (event: KeyboardEvent): boolean =>
  !event.ctrlKey &&
  !event.metaKey &&
  !event.altKey &&
  typeof Element !== 'undefined' &&
  event.target instanceof Element &&
  ((NATIVE_CONTROL_KEYS.has(event.key) &&
    event.target.closest('input[type="radio"], input[type="range"], [role="radiogroup"], [role="slider"]') !== null) ||
    (ACTIVATION_KEYS.has(event.key) && event.target.matches(ACTIVATABLE_CONTROL_SELECTOR)));

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
