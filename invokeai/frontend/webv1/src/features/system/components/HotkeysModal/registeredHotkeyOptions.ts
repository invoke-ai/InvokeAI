import type { Options } from 'react-hotkeys-hook';

/**
 * Adds the canvas text-session guard without converting a statically disabled hotkey into a mounted listener.
 *
 * react-hotkeys-hook does not attach its listener when `enabled` is exactly `false`. If `false` is wrapped in a
 * function instead, the listener remains attached and stops immediate propagation when the key matches. That can
 * prevent another, enabled handler for the same key from running.
 */
export const getRegisteredHotkeyOptions = (
  options: Options,
  isUncommittedCanvasTextSessionActive: () => boolean
): Options => {
  if (options.enabled === false) {
    return options;
  }

  const configuredEnabled = options.enabled;

  return {
    ...options,
    enabled: (event, hotkeysEvent) => {
      if (isUncommittedCanvasTextSessionActive()) {
        return false;
      }
      if (typeof configuredEnabled === 'function') {
        return configuredEnabled(event, hotkeysEvent);
      }
      return configuredEnabled ?? true;
    },
  };
};
