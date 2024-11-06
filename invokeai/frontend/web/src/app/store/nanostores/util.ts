import type { ReadableAtom } from 'nanostores';
import { atom } from 'nanostores';

/**
 * A fallback non-writable atom that always returns `false`, used when a nanostores atom is only conditionally available
 * in a hook or component.
 */
export const $false: ReadableAtom<boolean> = atom(false);
/**
 * A fallback non-writable atom that always returns `true`, used when a nanostores atom is only conditionally available
 * in a hook or component.
 */
export const $true: ReadableAtom<boolean> = atom(true);
