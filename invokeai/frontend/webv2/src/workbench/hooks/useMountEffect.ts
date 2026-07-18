import { useEffect, type EffectCallback } from 'react';

/** Explicit escape hatch for one-time external registration. */
export const useMountEffect = (effect: EffectCallback): void => {
  // The empty dependency list is the contract of this hook.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(effect, []);
};
