/** Checker tokens resolved through probe colors; `DEFAULT_CHECKER_COLORS` without a DOM or usable tokens. */

import { system } from '@theme/system';
import { type CheckerColors, DEFAULT_CHECKER_COLORS } from '@workbench/canvas-engine/api';

export const CHECKER_TOKEN_A = 'bg.inset';
export const CHECKER_TOKEN_B = 'bg.subtle';

/** The `var(--chakra-colors-…)` reference for a semantic color token, or `null` if unknown. */
const cssVarRef = (token: string): string | null => {
  const varName = system.tokens.getByName(`colors.${token}`)?.extensions.cssVar?.var;
  return varName ? `var(${varName})` : null;
};

/** Non-empty and not fully transparent, which is what a browser returns when a var fails to resolve. */
export const isUsableColor = (value: string | null | undefined): value is string =>
  typeof value === 'string' && value.trim() !== '' && value !== 'transparent' && value !== 'rgba(0, 0, 0, 0)';

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
