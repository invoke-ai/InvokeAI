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

/**
 * Text of each node failing `color-contrast`, sorted. Pinning known offenders by text keeps the rule on: a new
 * failing node, or a fixed one, changes the list.
 */
export const contrastOffenderTexts = (violations: readonly axe.Result[]): string[] =>
  violations
    .filter((violation) => violation.id === 'color-contrast')
    .flatMap((violation) =>
      violation.nodes.map((node) => document.querySelector(String(node.target[0]))?.textContent ?? node.html)
    )
    .sort();
