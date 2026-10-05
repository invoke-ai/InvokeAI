import axe from 'axe-core';

import { settleAnimations } from './settleAnimations.testing';

/**
 * Settle animations before axe to avoid transient contrast failures; return violations so callers choose
 * assertions. Architecture checks enforce this entry point.
 */
export const auditAccessibility = async (target: Element): Promise<axe.Result[]> => {
  await settleAnimations();

  return (await axe.run(target)).violations;
};
