export const IS_MAC_OS =
  typeof navigator !== 'undefined' &&
  (
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ??
    navigator.platform ??
    ''
  )
    .toLowerCase()
    .includes('mac');

const HOTKEY_KEY_ALIASES_BY_CODE = {
  Backquote: 'backquote',
  Minus: 'minus',
  Equal: 'equal',
  BracketLeft: 'bracketleft',
  BracketRight: 'bracketright',
  Backslash: 'backslash',
  Semicolon: 'semicolon',
  Quote: 'quote',
  Comma: 'comma',
  Period: 'period',
  Slash: 'slash',
} as const;

const HOTKEY_KEY_DISPLAY_ALIASES = {
  backquote: '`',
  minus: '-',
  equal: '=',
  bracketleft: '[',
  bracketright: ']',
  backslash: '\\',
  semicolon: ';',
  quote: "'",
  comma: ',',
  period: '.',
  slash: '/',
} as const;

export const getHotkeyKeyFromEvent = (key: string, code?: string): string => {
  if (code && code in HOTKEY_KEY_ALIASES_BY_CODE) {
    return HOTKEY_KEY_ALIASES_BY_CODE[code as keyof typeof HOTKEY_KEY_ALIASES_BY_CODE];
  }

  return key;
};

export const normalizeHotkeyKey = (key: string, isMac: boolean = IS_MAC_OS): string => {
  const keyMap: Record<string, string> = isMac
    ? {
        Meta: 'mod',
        Command: 'mod',
        Control: 'ctrl',
        Alt: 'alt',
        Shift: 'shift',
        ' ': 'space',
        '`': 'backquote',
        '-': 'minus',
        '=': 'equal',
        '[': 'bracketleft',
        ']': 'bracketright',
        '\\': 'backslash',
        ';': 'semicolon',
        "'": 'quote',
        ',': 'comma',
        '.': 'period',
        '/': 'slash',
      }
    : {
        Control: 'mod',
        Meta: 'meta',
        Alt: 'alt',
        Shift: 'shift',
        ' ': 'space',
        '`': 'backquote',
        '-': 'minus',
        '=': 'equal',
        '[': 'bracketleft',
        ']': 'bracketright',
        '\\': 'backslash',
        ';': 'semicolon',
        "'": 'quote',
        ',': 'comma',
        '.': 'period',
        '/': 'slash',
      };

  return keyMap[key] || key.toLowerCase();
};

export const canonicalizeHotkeyString = (hotkey: string, isMac: boolean = IS_MAC_OS): string => {
  return hotkey
    .split('+')
    .map((key) => normalizeHotkeyKey(key, isMac))
    .join('+');
};

export const formatHotkeyKeyForDisplay = (key: string, isMac: boolean = IS_MAC_OS): string => {
  const normalizedKey = key.toLowerCase();
  const displayKey =
    HOTKEY_KEY_DISPLAY_ALIASES[normalizedKey as keyof typeof HOTKEY_KEY_DISPLAY_ALIASES] ?? normalizedKey;

  if (isMac) {
    return displayKey.replaceAll('mod', 'cmd').replaceAll('alt', 'option');
  }

  return displayKey.replaceAll('mod', 'ctrl');
};

export const formatHotkeyStringForPlatform = (hotkey: string, isMac: boolean = IS_MAC_OS): string[] => {
  return hotkey.split('+').map((key) => formatHotkeyKeyForDisplay(key, isMac));
};
