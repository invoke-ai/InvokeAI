import type { HotkeyKeyboardLayoutMap } from 'features/system/components/HotkeysModal/hotkeyStrings';
import { useEffect, useState } from 'react';

type NavigatorWithKeyboardLayoutMap = Navigator & {
  keyboard?: {
    getLayoutMap?: () => Promise<HotkeyKeyboardLayoutMap>;
  };
};

let cachedKeyboardLayoutMap: HotkeyKeyboardLayoutMap | null | undefined;
let keyboardLayoutMapPromise: Promise<HotkeyKeyboardLayoutMap | null> | null = null;

const loadKeyboardLayoutMap = async (): Promise<HotkeyKeyboardLayoutMap | null> => {
  if (cachedKeyboardLayoutMap !== undefined) {
    return cachedKeyboardLayoutMap;
  }

  if (typeof navigator === 'undefined') {
    cachedKeyboardLayoutMap = null;
    return cachedKeyboardLayoutMap;
  }

  const keyboard = (navigator as NavigatorWithKeyboardLayoutMap).keyboard;
  if (!keyboard?.getLayoutMap) {
    cachedKeyboardLayoutMap = null;
    return cachedKeyboardLayoutMap;
  }

  try {
    cachedKeyboardLayoutMap = await keyboard.getLayoutMap();
  } catch {
    cachedKeyboardLayoutMap = null;
  }

  return cachedKeyboardLayoutMap;
};

export const useKeyboardLayoutMap = (): HotkeyKeyboardLayoutMap | null => {
  const [keyboardLayoutMap, setKeyboardLayoutMap] = useState<HotkeyKeyboardLayoutMap | null>(
    cachedKeyboardLayoutMap ?? null
  );

  useEffect(() => {
    let isSubscribed = true;

    if (!keyboardLayoutMapPromise) {
      keyboardLayoutMapPromise = loadKeyboardLayoutMap();
    }

    keyboardLayoutMapPromise.then((layoutMap) => {
      if (isSubscribed) {
        setKeyboardLayoutMap(layoutMap);
      }
    });

    return () => {
      isSubscribed = false;
    };
  }, []);

  return keyboardLayoutMap;
};
