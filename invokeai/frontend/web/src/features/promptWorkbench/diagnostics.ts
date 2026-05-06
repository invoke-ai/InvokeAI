import type { BaseModelType } from 'features/nodes/types/common';
import type { WildcardIndexItem } from 'services/api/endpoints/utilities';

import { getPromptModelCapabilities } from './modelCapabilities';
import { getMissingWildcardReferences, getWildcardReferences } from './wildcards';

const NUMERIC_ATTENTION_PATTERN = /\([^()\r\n]+:[+-]?\d+(?:\.\d+)?\)/;
const COMPEL_ATTENTION_PATTERN = /\)(?:[+-]+|[+-]?\d+(?:\.\d+)?)/;
const BRACKET_ATTENTION_PATTERN = /\[[^\]\r\n]+\]/;
const DYNAMIC_PROMPT_PATTERN = /\{[\s\S]*\}|\$\{[\s\S]*\}|__[^\r\n]+?__/;
const CYCLIC_WILDCARD_PATTERN = /__@[^\r\n_][^\r\n]*?__/;

export type PromptDiagnosticSeverity = 'ok' | 'info' | 'warning' | 'error';

export type PromptDiagnostic = {
  code: string;
  label: string;
  severity: PromptDiagnosticSeverity;
  description: string;
};

type GetPromptDiagnosticsArg = {
  prompt: string;
  modelBase: BaseModelType | null | undefined;
  wildcards: WildcardIndexItem[];
  wildcardIndexUnavailable?: boolean;
  wildcardIndexErrorCount: number;
  dynamicPromptCount: number;
  dynamicPromptMode: 'random' | 'combinatorial';
  dynamicPromptRandomRefreshMode?: 'manual' | 'per_enqueue';
  dynamicPromptError?: string | null;
};

export const getPromptDiagnostics = ({
  prompt,
  modelBase,
  wildcards,
  wildcardIndexUnavailable = false,
  wildcardIndexErrorCount,
  dynamicPromptCount,
  dynamicPromptMode,
  dynamicPromptRandomRefreshMode = 'manual',
  dynamicPromptError,
}: GetPromptDiagnosticsArg): PromptDiagnostic[] => {
  const capabilities = getPromptModelCapabilities(modelBase);
  const hasAttentionSyntax = getHasAttentionSyntax(prompt);
  const wildcardReferences = getWildcardReferences(prompt);
  const missingWildcards = wildcardIndexUnavailable ? [] : getMissingWildcardReferences(prompt, wildcards);
  const diagnostics: PromptDiagnostic[] = [
    {
      code: 'attention-support',
      label: capabilities.supportsAttentionWeights ? 'Weights OK' : 'Weights warn',
      severity: capabilities.supportsAttentionWeights ? 'ok' : hasAttentionSyntax ? 'warning' : 'info',
      description: capabilities.attentionWeightsLabel,
    },
  ];

  if (hasAttentionSyntax && !capabilities.supportsAttentionWeights) {
    diagnostics.push({
      code: 'attention-unsupported',
      label: 'Unsupported weights',
      severity: 'warning',
      description: 'This prompt uses weight syntax, but the selected model may encode it as literal text.',
    });
  }

  if (wildcardIndexUnavailable) {
    diagnostics.push({
      code: 'wildcards-unavailable',
      label: 'Wildcard error',
      severity: 'error',
      description: 'The local wildcard index could not be loaded. Restart the backend so /api/v1/utilities/wildcards is available.',
    });
  } else if (missingWildcards.length > 0) {
    diagnostics.push({
      code: 'wildcards-missing',
      label: `Missing ${missingWildcards.length}`,
      severity: 'error',
      description: `Missing wildcard${missingWildcards.length === 1 ? '' : 's'}: ${missingWildcards.join(', ')}`,
    });
  } else if (wildcardReferences.length > 0) {
    diagnostics.push({
      code: 'wildcards-found',
      label: `Wildcards ${wildcardReferences.length}`,
      severity: 'ok',
      description: 'All referenced wildcards are available locally.',
    });
  } else {
    diagnostics.push({
      code: 'wildcards-available',
      label: 'Wildcards',
      severity: 'info',
      description: `${wildcards.length} local wildcard${wildcards.length === 1 ? '' : 's'} available. Type __ to insert one.`,
    });
  }

  if (!wildcardIndexUnavailable && wildcardIndexErrorCount > 0) {
    diagnostics.push({
      code: 'wildcards-index-errors',
      label: `Index errors ${wildcardIndexErrorCount}`,
      severity: 'warning',
      description: 'Some local wildcard files could not be read.',
    });
  }

  if (dynamicPromptError) {
    diagnostics.push({
      code: 'dynamic-error',
      label: 'Dynamic error',
      severity: 'error',
      description: `Dynamic prompt parser error: ${dynamicPromptError}`,
    });
  } else if (getHasDynamicPromptSyntax(prompt)) {
    const count = Math.max(dynamicPromptCount, 1);
    const isCombinatorial = dynamicPromptMode === 'combinatorial';
    const hasCyclicWildcard = getHasCyclicWildcardSyntax(prompt);
    diagnostics.push({
      code: 'dynamic-active',
      label: getDynamicPromptLabel({
        count,
        isCombinatorial,
        hasCyclicWildcard,
        randomRefreshMode: dynamicPromptRandomRefreshMode,
      }),
      severity: 'ok',
      description: getDynamicPromptDescription({
        isCombinatorial,
        hasCyclicWildcard,
        randomRefreshMode: dynamicPromptRandomRefreshMode,
      }),
    });
  }

  return diagnostics;
};

export const getHasAttentionSyntax = (prompt: string): boolean =>
  NUMERIC_ATTENTION_PATTERN.test(prompt) || COMPEL_ATTENTION_PATTERN.test(prompt) || BRACKET_ATTENTION_PATTERN.test(prompt);

export const getHasDynamicPromptSyntax = (prompt: string): boolean => DYNAMIC_PROMPT_PATTERN.test(prompt);

export const getHasCyclicWildcardSyntax = (prompt: string): boolean => CYCLIC_WILDCARD_PATTERN.test(prompt);

const getDynamicPromptLabel = (arg: {
  count: number;
  isCombinatorial: boolean;
  hasCyclicWildcard: boolean;
  randomRefreshMode: 'manual' | 'per_enqueue';
}): string => {
  const { count, isCombinatorial, hasCyclicWildcard, randomRefreshMode } = arg;

  if (isCombinatorial) {
    return `All ${count}`;
  }

  if (hasCyclicWildcard) {
    return `Cycle ${count}`;
  }

  if (randomRefreshMode === 'per_enqueue') {
    return `Random ${count}/run`;
  }

  return `Random ${count}`;
};

const getDynamicPromptDescription = (arg: {
  isCombinatorial: boolean;
  hasCyclicWildcard: boolean;
  randomRefreshMode: 'manual' | 'per_enqueue';
}): string => {
  const { isCombinatorial, hasCyclicWildcard, randomRefreshMode } = arg;

  if (isCombinatorial) {
    return 'All-combinations prompt expansion is active for this prompt.';
  }

  if (hasCyclicWildcard) {
    return 'Cyclic wildcard expansion is deterministic. Use Random to roll one value on each Invoke.';
  }

  if (randomRefreshMode === 'per_enqueue') {
    return 'Random prompt sampling will roll fresh values when generation is queued.';
  }

  return 'Random prompt sampling is fixed to the current preview until Reshuffle is used.';
};
