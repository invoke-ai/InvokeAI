export const IS_MAC_OS =
  typeof navigator !== 'undefined' &&
  (
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ??
    navigator.platform ??
    ''
  )
    .toLowerCase()
    .includes('mac');

export type HotkeyKeyboardLayoutMap = Pick<ReadonlyMap<string, string>, 'get'>;

const HOTKEY_PHYSICAL_KEY_ALIASES = [
  { code: 'Backquote', token: 'backquote', display: '`', shiftedDisplay: '~' },
  { code: 'Minus', token: 'minus', display: '-', shiftedDisplay: '_' },
  { code: 'Equal', token: 'equal', display: '=', shiftedDisplay: '+' },
  { code: 'BracketLeft', token: 'bracketleft', display: '[', shiftedDisplay: '{' },
  { code: 'BracketRight', token: 'bracketright', display: ']', shiftedDisplay: '}' },
  { code: 'Backslash', token: 'backslash', display: '\\', shiftedDisplay: '|' },
  { code: 'Semicolon', token: 'semicolon', display: ';', shiftedDisplay: ':' },
  { code: 'Quote', token: 'quote', display: "'", shiftedDisplay: '"' },
  { code: 'Comma', token: 'comma', display: ',', shiftedDisplay: '<' },
  { code: 'Period', token: 'period', display: '.', shiftedDisplay: '>' },
  { code: 'Slash', token: 'slash', display: '/', shiftedDisplay: '?' },
] as const;

const HOTKEY_KEY_ALIASES_BY_CODE = Object.fromEntries(
  HOTKEY_PHYSICAL_KEY_ALIASES.map(({ code, token }) => [code, token])
) as Record<string, string>;

const HOTKEY_KEY_CODES_BY_ALIAS = Object.fromEntries(
  HOTKEY_PHYSICAL_KEY_ALIASES.map(({ code, token }) => [token, code])
) as Record<string, string>;

const HOTKEY_KEY_DISPLAY_ALIASES = Object.fromEntries(
  HOTKEY_PHYSICAL_KEY_ALIASES.map(({ token, display }) => [token, display])
) as Record<string, string>;

const HOTKEY_PHYSICAL_KEY_ALIASES_BY_KEY = Object.fromEntries(
  HOTKEY_PHYSICAL_KEY_ALIASES.flatMap((alias) => [
    [alias.token, alias],
    [alias.display, alias],
    [alias.shiftedDisplay, alias],
  ])
) as Record<string, (typeof HOTKEY_PHYSICAL_KEY_ALIASES)[number]>;

const HOTKEY_MODIFIER_KEYS = new Set(['mod', 'ctrl', 'meta', 'shift', 'alt']);

export const getHotkeyKeyFromEvent = (key: string, code?: string): string => {
  const codeAlias = code ? HOTKEY_KEY_ALIASES_BY_CODE[code] : undefined;
  if (codeAlias) {
    return codeAlias;
  }

  return key;
};

export const normalizeHotkeyKey = (key: string, isMac: boolean = IS_MAC_OS): string => {
  const keyMap: Record<string, string> = isMac
    ? {
        Meta: 'mod',
        meta: 'mod',
        Command: 'mod',
        command: 'mod',
        Control: 'ctrl',
        control: 'ctrl',
        Alt: 'alt',
        alt: 'alt',
        Shift: 'shift',
        shift: 'shift',
        ' ': 'space',
        Spacebar: 'space',
        spacebar: 'space',
      }
    : {
        Control: 'mod',
        control: 'mod',
        Meta: 'meta',
        meta: 'meta',
        Alt: 'alt',
        alt: 'alt',
        Shift: 'shift',
        shift: 'shift',
        ' ': 'space',
        Spacebar: 'space',
        spacebar: 'space',
      };

  return keyMap[key] || key.toLowerCase();
};

export const canonicalizeHotkeyString = (hotkey: string, isMac: boolean = IS_MAC_OS): string => {
  return hotkey
    .split('+')
    .map((key) => normalizeHotkeyKey(key.trim() || key, isMac))
    .join('+');
};

export const getHotkeyStringAliases = (hotkey: string, isMac: boolean = IS_MAC_OS): string[] => {
  const parts = canonicalizeHotkeyString(hotkey, isMac).split('+');
  const regularKeyIndex = parts.findIndex((part) => !HOTKEY_MODIFIER_KEYS.has(part));

  if (regularKeyIndex === -1) {
    return [parts.join('+')];
  }

  const regularKey = parts[regularKeyIndex];
  if (!regularKey) {
    return [parts.join('+')];
  }

  const modifiers = parts.filter((_, index) => index !== regularKeyIndex);
  const physicalKeyAlias = HOTKEY_PHYSICAL_KEY_ALIASES_BY_KEY[regularKey];

  if (!physicalKeyAlias) {
    return [parts.join('+')];
  }

  const regularKeyAliases = new Set<string>([regularKey, physicalKeyAlias.token, physicalKeyAlias.display]);
  if (modifiers.includes('shift')) {
    regularKeyAliases.add(physicalKeyAlias.shiftedDisplay);
  }

  return [...regularKeyAliases].map((alias) => [...modifiers, alias].join('+'));
};

export const areHotkeyStringsEquivalent = (
  firstHotkey: string,
  secondHotkey: string,
  isMac: boolean = IS_MAC_OS
): boolean => {
  const firstAliases = new Set(getHotkeyStringAliases(firstHotkey, isMac));
  return getHotkeyStringAliases(secondHotkey, isMac).some((alias) => firstAliases.has(alias));
};

export const formatHotkeyKeyForDisplay = (
  key: string,
  isMac: boolean = IS_MAC_OS,
  keyboardLayoutMap?: HotkeyKeyboardLayoutMap | null
): string => {
  const normalizedKey = key.toLowerCase();
  const layoutMapKey = keyboardLayoutMap?.get(HOTKEY_KEY_CODES_BY_ALIAS[normalizedKey] ?? '');
  const displayKey = layoutMapKey || HOTKEY_KEY_DISPLAY_ALIASES[normalizedKey] || normalizedKey;

  if (isMac) {
    return displayKey.replaceAll('mod', 'cmd').replaceAll('alt', 'option');
  }

  return displayKey.replaceAll('mod', 'ctrl');
};

export const formatHotkeyStringForPlatform = (
  hotkey: string,
  isMac: boolean = IS_MAC_OS,
  keyboardLayoutMap?: HotkeyKeyboardLayoutMap | null
): string[] => {
  return hotkey.split('+').map((key) => formatHotkeyKeyForDisplay(key, isMac, keyboardLayoutMap));
};
