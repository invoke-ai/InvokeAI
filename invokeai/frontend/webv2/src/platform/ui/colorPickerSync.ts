import { normalizeHex } from './color';

/**
 * True when `externalValue` (the latest `value` prop given to `ColorPicker`)
 * should be re-parsed into the picker's internal `Color`, versus kept as-is.
 *
 * We only want to re-sync from the external string when it reflects a change
 * the consumer made independently of the picker (e.g. a programmatic reset) —
 * not the round trip of the value the picker *just emitted itself* via
 * `onValueChange`. That round trip loses hue information for any grey/black/
 * white color (saturation or value = 0, where hex/RGB is hue-agnostic), so
 * re-parsing it on every render would snap the hue thumb back to whatever
 * `parseColor` yields for an indeterminate hue — the picker would only ever
 * be able to *emit* a new hue at S=0, never actually keep it.
 *
 * Values are compared after normalization, so a consumer that stores or echoes
 * back a differently-cased or shorthand form of the same color (`#FF0000` for
 * `#ff0000`, `#f00` for `#ff0000`, `#ff0000ff` for `#ff0000`) does not count as
 * an independent change and does not clobber the internal hue.
 */
export const shouldSyncExternalColor = (
  externalValue: string,
  previousExternalValue: string,
  lastEmittedValue: string
): boolean => {
  const external = normalizeHex(externalValue, externalValue);

  return (
    external !== normalizeHex(previousExternalValue, previousExternalValue) &&
    external !== normalizeHex(lastEmittedValue, lastEmittedValue)
  );
};
