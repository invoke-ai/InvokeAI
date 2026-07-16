/**
 * Resolves the transparency checkerboard's two square colors from the live Chakra
 * theme so the infinite-canvas surround uses theme tokens (not hardcoded greys),
 * and re-resolves on theme / color-mode switches. The resolved colors are fed
 * one-directionally into the engine's `checkerColors` store (the same
 * settings→engine-store pattern as the boolean canvas settings); the engine stays
 * React-free and rebuilds its cached checker tile when the colors change.
 *
 * Resolution reads the *computed* backgroundColor of a hidden probe element set to
 * each semantic token's CSS custom property — the browser's ground truth for the
 * active `<html data-theme>`. Falls back to {@link DEFAULT_CHECKER_COLORS} when the
 * DOM is unavailable (node tests) or a token yields nothing usable.
 */

import { system } from '@theme/system';
import { type CheckerColors, DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/render/compositor';

/**
 * The two Chakra semantic tokens used for the checker squares. `bg.subtle` and
 * `bg.emphasized` are two adjacent neutral surfaces, giving a low-contrast checker
 * that reads as "empty" in every theme (light and dark) without a bespoke pair.
 */
export const CHECKER_TOKEN_A = 'bg.subtle';
export const CHECKER_TOKEN_B = 'bg.emphasized';

/** The `var(--chakra-colors-…)` reference for a semantic color token, or `null` if unknown. */
const cssVarRef = (token: string): string | null => {
  const varName = system.tokens.getByName(`colors.${token}`)?.extensions.cssVar?.var;
  return varName ? `var(${varName})` : null;
};

/**
 * Whether a computed color string is usable — a non-empty color that isn't fully
 * transparent (the browser's answer when a var failed to resolve).
 */
export const isUsableColor = (value: string | null | undefined): value is string =>
  typeof value === 'string' && value.trim() !== '' && value !== 'transparent' && value !== 'rgba(0, 0, 0, 0)';

/** Returns `resolved` when usable, else `fallback`. */
export const pickCheckerColor = (resolved: string | null | undefined, fallback: string): string =>
  isUsableColor(resolved) ? resolved : fallback;

/**
 * Resolves both checker colors from the current theme. Safe to call anytime; in a
 * non-DOM environment (or before the theme is applied) it returns the fallback
 * pair so callers always get concrete colors.
 */
export const resolveCheckerColors = (): CheckerColors => {
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function' || !document.body) {
    return { ...DEFAULT_CHECKER_COLORS };
  }
  const probe = document.createElement('div');
  probe.style.display = 'none';
  document.body.appendChild(probe);
  try {
    const read = (token: string, fallback: string): string => {
      const ref = cssVarRef(token);
      if (!ref) {
        return fallback;
      }
      probe.style.backgroundColor = ref;
      return pickCheckerColor(getComputedStyle(probe).backgroundColor, fallback);
    };
    return {
      a: read(CHECKER_TOKEN_A, DEFAULT_CHECKER_COLORS.a),
      b: read(CHECKER_TOKEN_B, DEFAULT_CHECKER_COLORS.b),
    };
  } finally {
    probe.remove();
  }
};
