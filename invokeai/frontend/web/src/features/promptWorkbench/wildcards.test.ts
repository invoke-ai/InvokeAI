import { describe, expect, it } from 'vitest';

import {
  applyWildcardCompletion,
  filterWildcardOptions,
  getCyclicWildcardToken,
  getMissingWildcardReferences,
  getWildcardAutocompleteStatusMessage,
  getWildcardCompletionContext,
  getWildcardDisplayPath,
} from './wildcards';

const wildcards = [
  {
    token: '__camera/lens__',
    path: 'camera/lens',
    label: 'lens',
    file_type: 'txt' as const,
    value_count: 2,
    samples: ['50mm', '85mm'],
  },
  {
    token: '__lighting/studio__',
    path: 'lighting/studio',
    label: 'studio',
    file_type: 'yaml' as const,
    value_count: 1,
    samples: ['softbox'],
  },
];

describe('prompt workbench wildcards', () => {
  it('detects wildcard autocomplete context after double underscores', () => {
    expect(getWildcardCompletionContext('portrait __camera', 17)).toEqual({
      start: 9,
      end: 17,
      query: 'camera',
    });
  });

  it('does not detect autocomplete context after a closed wildcard', () => {
    expect(getWildcardCompletionContext('portrait __camera/lens__', 24)).toBeNull();
  });

  it('filters wildcard options by path and samples', () => {
    expect(filterWildcardOptions(wildcards, 'soft').map((wildcard) => wildcard.path)).toEqual(['lighting/studio']);
    expect(filterWildcardOptions(wildcards, 'lens').map((wildcard) => wildcard.path)).toEqual(['camera/lens']);
  });

  it('displays wildcard paths without syntax delimiters', () => {
    expect(getWildcardDisplayPath(wildcards[0]!)).toBe('camera/lens');
  });

  it('explains why autocomplete has no visible options', () => {
    expect(
      getWildcardAutocompleteStatusMessage({
        isLoading: false,
        isUnavailable: true,
        optionCount: 0,
        query: 'camera',
        wildcardCount: 0,
      })
    ).toBe('Wildcard index unavailable. Restart the backend or check the wildcard endpoint.');

    expect(
      getWildcardAutocompleteStatusMessage({
        isLoading: false,
        isUnavailable: false,
        optionCount: 0,
        query: 'missing',
        wildcardCount: wildcards.length,
      })
    ).toBe('No local wildcards match "missing".');
  });

  it('reports missing wildcard references while allowing glob references', () => {
    expect(getMissingWildcardReferences('__camera/*__ __missing__', wildcards)).toEqual(['missing']);
  });

  it('applies wildcard, cyclic wildcard, and fixed-value completions', () => {
    const context = getWildcardCompletionContext('portrait __camera', 17);

    expect(context).not.toBeNull();
    expect(applyWildcardCompletion('portrait __camera', context!, '__camera/lens__')).toEqual({
      prompt: 'portrait __camera/lens__',
      caret: 24,
    });
    expect(applyWildcardCompletion('portrait __camera', context!, getCyclicWildcardToken('camera/lens')).prompt).toBe(
      'portrait __@camera/lens__'
    );
    expect(applyWildcardCompletion('portrait __camera', context!, '85mm portrait lens').prompt).toBe(
      'portrait 85mm portrait lens'
    );
  });
});
