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

export const eventToHotkeyString = (event: KeyboardEvent): string => {
  if (event.isComposing || event.keyCode === 229 || ['Alt', 'Control', 'Meta', 'Shift'].includes(event.key)) {
    return '';
  }

  const modifiers: string[] = [];

  if (event.metaKey || event.ctrlKey) {
    modifiers.push(IS_MAC_OS ? (event.metaKey ? 'mod' : 'ctrl') : event.ctrlKey ? 'mod' : 'meta');
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

export const formatHotkeyForPlatform = (hotkey: string): string[] =>
  normalizeHotkeyString(hotkey)
    .split('+')
    .filter(Boolean)
    .map((part) => (IS_MAC_OS ? part.replace('mod', 'cmd').replace('alt', 'option') : part.replace('mod', 'ctrl')));

export const isEditableHotkeyTarget = (target: EventTarget | null): boolean => {
  if (typeof HTMLElement === 'undefined') {
    return false;
  }

  if (!(target instanceof HTMLElement)) {
    return false;
  }

  return Boolean(target.closest('input, textarea, select, [contenteditable="true"]'));
};
