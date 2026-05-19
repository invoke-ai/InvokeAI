import type { DynamicPromptRandomRefreshMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import {
  getHasCyclicWildcardSyntax,
  getHasDynamicPromptSyntax,
  getHasMixedCyclicAndNonCyclicDynamicPromptSyntax,
} from 'features/dynamicPrompts/util/promptIntent';
import type { BaseModelType } from 'features/nodes/types/common';
import type { WildcardIndexItem } from 'services/api/endpoints/utilities';

import { type PromptWorkbenchTranslation, tx } from './i18n';
import { getPromptModelCapabilities } from './modelCapabilities';
import { getMissingWildcardReferences, getWildcardReferences } from './wildcards';

const NUMERIC_ATTENTION_PATTERN = /\([^()\r\n]+:[+-]?\d+(?:\.\d+)?\)/;
const COMPEL_ATTENTION_PATTERN = /\)(?:[+-]+|[+-]?\d+(?:\.\d+)?)/;
const BRACKET_ATTENTION_PATTERN = /\[[^\]\r\n]+\]/;

export type PromptDiagnosticSeverity = 'ok' | 'info' | 'warning' | 'error';

export type PromptDiagnostic = {
  code: string;
  label: PromptWorkbenchTranslation;
  severity: PromptDiagnosticSeverity;
  description: PromptWorkbenchTranslation;
};

type GetPromptDiagnosticsArg = {
  prompt: string;
  modelBase: BaseModelType | null | undefined;
  wildcards: WildcardIndexItem[];
  wildcardIndexUnavailable?: boolean;
  wildcardIndexErrorCount: number;
  dynamicPromptCount: number;
  dynamicPromptMode: 'random' | 'combinatorial';
  dynamicPromptRandomRefreshMode?: DynamicPromptRandomRefreshMode;
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
  dynamicPromptRandomRefreshMode = 'per_image',
  dynamicPromptError,
}: GetPromptDiagnosticsArg): PromptDiagnostic[] => {
  const capabilities = getPromptModelCapabilities(modelBase);
  const hasAttentionSyntax = getHasAttentionSyntax(prompt);
  const wildcardReferences = getWildcardReferences(prompt);
  const missingWildcards = wildcardIndexUnavailable ? [] : getMissingWildcardReferences(prompt, wildcards);
  const diagnostics: PromptDiagnostic[] = [];

  if (hasAttentionSyntax && !capabilities.supportsAttentionWeights) {
    diagnostics.push({
      code: 'attention-unsupported',
      label: tx('promptWorkbench.diagnostics.weightsLiteralLabel'),
      severity: 'warning',
      description: tx('promptWorkbench.diagnostics.weightsLiteralDesc'),
    });
  }

  if (wildcardIndexUnavailable) {
    diagnostics.push({
      code: 'wildcards-unavailable',
      label: tx('promptWorkbench.diagnostics.wildcardErrorLabel'),
      severity: 'error',
      description: tx('promptWorkbench.diagnostics.wildcardErrorDesc'),
    });
  } else if (missingWildcards.length > 0) {
    diagnostics.push({
      code: 'wildcards-missing',
      label: tx('promptWorkbench.diagnostics.missingLabel', { count: missingWildcards.length }),
      severity: 'error',
      description: tx('promptWorkbench.diagnostics.missingDesc', {
        count: missingWildcards.length,
        wildcards: missingWildcards.join(', '),
      }),
    });
  } else if (wildcardReferences.length > 0) {
    diagnostics.push({
      code: 'wildcards-found',
      label: tx('promptWorkbench.diagnostics.wildcardsFoundLabel', { count: wildcardReferences.length }),
      severity: 'ok',
      description: tx('promptWorkbench.diagnostics.wildcardsFoundDesc'),
    });
  }

  if (!wildcardIndexUnavailable && wildcardIndexErrorCount > 0) {
    diagnostics.push({
      code: 'wildcards-index-errors',
      label: tx('promptWorkbench.diagnostics.indexErrorsLabel', { count: wildcardIndexErrorCount }),
      severity: 'warning',
      description: tx('promptWorkbench.diagnostics.indexErrorsDesc'),
    });
  }

  if (dynamicPromptError) {
    diagnostics.push({
      code: 'dynamic-error',
      label: tx('promptWorkbench.diagnostics.dynamicErrorLabel'),
      severity: 'error',
      description: tx('promptWorkbench.diagnostics.dynamicErrorDesc', { error: dynamicPromptError }),
    });
  } else if (getHasDynamicPromptSyntax(prompt)) {
    const count = Math.max(dynamicPromptCount, 1);
    const isCombinatorial = dynamicPromptMode === 'combinatorial';
    const hasCyclicWildcard = getHasCyclicWildcardSyntax(prompt);
    const hasMixedCyclicAndNonCyclicSyntax = getHasMixedCyclicAndNonCyclicDynamicPromptSyntax(prompt);
    diagnostics.push({
      code: 'dynamic-active',
      label: getDynamicPromptLabel({
        count,
        isCombinatorial,
        hasCyclicWildcard,
        hasMixedCyclicAndNonCyclicSyntax,
        randomRefreshMode: dynamicPromptRandomRefreshMode,
      }),
      severity: 'ok',
      description: getDynamicPromptDescription({
        isCombinatorial,
        hasCyclicWildcard,
        hasMixedCyclicAndNonCyclicSyntax,
        randomRefreshMode: dynamicPromptRandomRefreshMode,
      }),
    });
  }

  return diagnostics;
};

export const getHasAttentionSyntax = (prompt: string): boolean =>
  NUMERIC_ATTENTION_PATTERN.test(prompt) ||
  COMPEL_ATTENTION_PATTERN.test(prompt) ||
  BRACKET_ATTENTION_PATTERN.test(prompt);

const getDynamicPromptLabel = (arg: {
  count: number;
  isCombinatorial: boolean;
  hasCyclicWildcard: boolean;
  hasMixedCyclicAndNonCyclicSyntax: boolean;
  randomRefreshMode: DynamicPromptRandomRefreshMode;
}): PromptWorkbenchTranslation => {
  const { count, isCombinatorial, hasCyclicWildcard, hasMixedCyclicAndNonCyclicSyntax, randomRefreshMode } = arg;

  if (isCombinatorial) {
    return tx('promptWorkbench.diagnostics.dynamicAllLabel', { count });
  }

  if (hasMixedCyclicAndNonCyclicSyntax) {
    return tx('promptWorkbench.diagnostics.dynamicMixedLabel');
  }

  if (hasCyclicWildcard) {
    return tx('promptWorkbench.diagnostics.dynamicCycleLabel', { count });
  }

  if (randomRefreshMode === 'per_image') {
    return tx('promptWorkbench.behavior.randomImageShort');
  }

  if (randomRefreshMode === 'per_enqueue') {
    return tx('promptWorkbench.behavior.randomInvokeShort');
  }

  return tx('promptWorkbench.behavior.randomPreviewLabel');
};

const getDynamicPromptDescription = (arg: {
  isCombinatorial: boolean;
  hasCyclicWildcard: boolean;
  hasMixedCyclicAndNonCyclicSyntax: boolean;
  randomRefreshMode: DynamicPromptRandomRefreshMode;
}): PromptWorkbenchTranslation => {
  const { isCombinatorial, hasCyclicWildcard, hasMixedCyclicAndNonCyclicSyntax, randomRefreshMode } = arg;

  if (isCombinatorial) {
    return tx('promptWorkbench.diagnostics.dynamicAllDesc');
  }

  if (hasMixedCyclicAndNonCyclicSyntax) {
    return tx('promptWorkbench.diagnostics.dynamicMixedDesc');
  }

  if (hasCyclicWildcard) {
    return tx('promptWorkbench.diagnostics.dynamicCycleDesc');
  }

  if (randomRefreshMode === 'per_image') {
    return tx('promptWorkbench.diagnostics.dynamicRandomImageDesc');
  }

  if (randomRefreshMode === 'per_enqueue') {
    return tx('promptWorkbench.diagnostics.dynamicRandomInvokeDesc');
  }

  return tx('promptWorkbench.diagnostics.dynamicRandomPreviewDesc');
};
