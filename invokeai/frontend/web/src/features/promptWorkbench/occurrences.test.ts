import { describe, expect, it } from 'vitest';

import {
  getPromptWeightOccurrences,
  getPromptWildcardOccurrences,
  getPromptWorkbenchOccurrences,
  getWildcardBehaviorLabel,
  removePromptRange,
  replacePromptRange,
} from './occurrences';

const wildcards = [
  {
    token: '__camera/lens__',
    path: 'camera/lens',
    label: 'lens',
    file_type: 'txt' as const,
    value_count: 4,
    samples: ['35mm lens', '50mm lens'],
  },
  {
    token: '__lighting/studio__',
    path: 'lighting/studio',
    label: 'studio',
    file_type: 'txt' as const,
    value_count: 3,
    samples: ['softbox'],
  },
];

describe('prompt workbench occurrences', () => {
  it('parses random wildcard occurrences', () => {
    const occurrences = getPromptWildcardOccurrences({
      prompt: 'portrait __camera/lens__',
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
    });

    expect(occurrences).toHaveLength(1);
    expect(occurrences[0]).toMatchObject({
      type: 'wildcard',
      path: 'camera/lens',
      behavior: 'random',
      valueCount: 4,
    });
    expect(getWildcardBehaviorLabel(occurrences[0]!, 'per_enqueue')).toBe('Random every Invoke');
  });

  it('parses cyclic wildcard occurrences', () => {
    expect(
      getPromptWildcardOccurrences({
        prompt: 'portrait __@camera/lens__',
        wildcards,
        wildcardIndexUnavailable: false,
        dynamicPromptMode: 'random',
      })[0]
    ).toMatchObject({
      path: 'camera/lens',
      behavior: 'cycle',
    });
  });

  it('marks unknown wildcards as missing', () => {
    expect(
      getPromptWildcardOccurrences({
        prompt: 'portrait __missing/path__',
        wildcards,
        wildcardIndexUnavailable: false,
        dynamicPromptMode: 'random',
      })[0]
    ).toMatchObject({
      path: 'missing/path',
      behavior: 'missing',
      valueCount: null,
    });
  });

  it('replaces a specific duplicate wildcard occurrence', () => {
    const prompt = '__camera/lens__, __camera/lens__';
    const occurrences = getPromptWildcardOccurrences({
      prompt,
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
    });

    expect(replacePromptRange(prompt, occurrences[1]!.range, '__@camera/lens__').prompt).toBe(
      '__camera/lens__, __@camera/lens__'
    );
  });

  it('removes wildcard tokens cleanly from comma-separated prompts', () => {
    const prompt = 'portrait, __camera/lens__, studio';
    const occurrence = getPromptWildcardOccurrences({
      prompt,
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
    })[0]!;

    expect(removePromptRange(prompt, occurrence.range).prompt).toBe('portrait, studio');
  });

  it('removes wildcard tokens cleanly when the comma follows the token', () => {
    const prompt = 'portrait __camera/lens__, studio';
    const occurrence = getPromptWildcardOccurrences({
      prompt,
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
    })[0]!;

    expect(removePromptRange(prompt, occurrence.range).prompt).toBe('portrait, studio');
  });

  it('detects weighted spans and unsupported model warnings', () => {
    expect(getPromptWeightOccurrences({ prompt: '(face)++ cat+', supportsAttentionWeights: false })).toMatchObject([
      { type: 'weight', text: '(face)++', attention: '++', isSupported: false },
      { type: 'weight', text: 'cat+', attention: '+', isSupported: false },
    ]);
  });

  it('returns mixed prompt workbench occurrences in prompt order', () => {
    expect(
      getPromptWorkbenchOccurrences({
        prompt: '(face)++ __camera/lens__',
        wildcards,
        wildcardIndexUnavailable: false,
        dynamicPromptMode: 'combinatorial',
        supportsAttentionWeights: true,
      }).map((occurrence) => occurrence.type)
    ).toEqual(['weight', 'wildcard']);
  });
});
