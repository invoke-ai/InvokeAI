/**
 * Resolve checker semantic tokens through computed probe colors and feed the engine one-way. Use {@link
 * DEFAULT_CHECKER_COLORS} without DOM or usable tokens; changed colors rebuild its cached tile.
 */

import { system } from '@theme/system';
import { type CheckerColors, DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/api';

export const CHECKER_TOKEN_A = 'bg.inset';
export const CHECKER_TOKEN_B = 'bg.subtle';

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

/** Return concrete theme checker colors, falling back when DOM/theme values are unavailable. */
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
