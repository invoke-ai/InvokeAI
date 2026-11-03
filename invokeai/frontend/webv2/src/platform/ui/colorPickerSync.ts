import { normalizeHex } from './color';

/**
 * Ignore only an outstanding normalized echo, preserving hue lost in hex for greys. Independent changes—including
 * a later return to the last emitted color—must resync.
 */
export const shouldSyncExternalColor = (
  externalValue: string,
  previousExternalValue: string,
  lastEmittedValue: string,
  isAwaitingEcho: boolean
): boolean => {
  const external = normalizeHex(externalValue, externalValue);

  if (external === normalizeHex(previousExternalValue, previousExternalValue)) {
    return false;
  }

  return !(isAwaitingEcho && external === normalizeHex(lastEmittedValue, lastEmittedValue));
};
