import { describe, expect, it } from 'vitest';

import {
  getPromptWeightOccurrences,
  getPromptWildcardOccurrences,
  getPromptWorkbenchOccurrences,
  getWeightShortLabel,
  getWildcardBehaviorActionIntent,
  getWildcardBehaviorIconType,
  getWildcardBehaviorLabel,
  getWildcardBehaviorShortLabel,
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
    expect(getWildcardBehaviorLabel(occurrences[0]!, 'per_image')).toBe('Random per Image');
    expect(getWildcardBehaviorLabel(occurrences[0]!, 'per_enqueue')).toBe('Random per Invoke');
    expect(getWildcardBehaviorShortLabel(occurrences[0]!, 'per_image')).toBe('Random/image');
    expect(getWildcardBehaviorShortLabel(occurrences[0]!, 'per_enqueue')).toBe('Random/invoke');
    expect(getWildcardBehaviorShortLabel(occurrences[0]!, 'manual')).toBe('Preview');
    expect(getWildcardBehaviorIconType(occurrences[0]!)).toBe('random');
  });

  it('parses cyclic wildcard occurrences', () => {
    const occurrence =
      getPromptWildcardOccurrences({
        prompt: 'portrait __@camera/lens__',
        wildcards,
        wildcardIndexUnavailable: false,
        dynamicPromptMode: 'random',
      })[0]!;

    expect(occurrence).toMatchObject({
      path: 'camera/lens',
      behavior: 'cycle',
    });
    expect(getWildcardBehaviorShortLabel(occurrence, 'per_image')).toBe('Cycle');
    expect(getWildcardBehaviorIconType(occurrence)).toBe('cycle');
  });

  it('marks unknown wildcards as missing', () => {
    const occurrence =
      getPromptWildcardOccurrences({
        prompt: 'portrait __missing/path__',
        wildcards,
        wildcardIndexUnavailable: false,
        dynamicPromptMode: 'random',
      })[0]!;

    expect(occurrence).toMatchObject({
      path: 'missing/path',
      behavior: 'missing',
      valueCount: null,
    });
    expect(getWildcardBehaviorShortLabel(occurrence, 'per_image')).toBe('Missing');
    expect(getWildcardBehaviorIconType(occurrence)).toBe('warning');
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

  it('returns compact wildcard labels for all-combinations prompts', () => {
    const occurrence = getPromptWildcardOccurrences({
      prompt: 'portrait __camera/lens__',
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'combinatorial',
    })[0]!;

    expect(occurrence.behavior).toBe('all');
    expect(getWildcardBehaviorShortLabel(occurrence, 'per_image')).toBe('All');
    expect(getWildcardBehaviorIconType(occurrence)).toBe('all');
  });

  it('returns compact weight labels for supported and unsupported weights', () => {
    const supported = getPromptWeightOccurrences({ prompt: '(face)++', supportsAttentionWeights: true });
    const unsupported = getPromptWeightOccurrences({ prompt: '(face)+', supportsAttentionWeights: false });

    expect(getWeightShortLabel(supported[0]!)).toBe('++');
    expect(getWeightShortLabel({ ...supported[0]!, attention: 1.2 })).toBe('1.2');
    expect(getWeightShortLabel(unsupported[0]!)).toBe('Literal?');
  });

  it('maps wildcard behavior menu actions to prompt intent', () => {
    expect(getWildcardBehaviorActionIntent('random', 'camera/lens')).toEqual({
      replacement: '__camera/lens__',
    });
    expect(getWildcardBehaviorActionIntent('cycle', 'camera/lens')).toEqual({
      replacement: '__@camera/lens__',
    });
    expect(getWildcardBehaviorActionIntent('fixed', 'camera/lens')).toEqual({ opensFixedValues: true });
    expect(getWildcardBehaviorActionIntent('remove', 'camera/lens')).toEqual({ removesPrompt: true });
  });

  it('merges wrapper-weighted wildcards into one wildcard occurrence', () => {
    const occurrences = getPromptWorkbenchOccurrences({
      prompt: '(__lighting/studio__)++, (face)+',
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
      supportsAttentionWeights: true,
    });

    expect(occurrences).toHaveLength(2);
    expect(occurrences[0]).toMatchObject({
      type: 'wildcard',
      path: 'lighting/studio',
      weight: {
        type: 'weight',
        text: '(__lighting/studio__)++',
        attention: '++',
        isSupported: true,
      },
    });
    expect(occurrences[1]).toMatchObject({
      type: 'weight',
      text: '(face)+',
    });
  });

  it('marks wrapper-weighted wildcards as unsupported when the model does not support weights', () => {
    const occurrences = getPromptWorkbenchOccurrences({
      prompt: '(__lighting/studio__)++',
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
      supportsAttentionWeights: false,
    });

    expect(occurrences[0]).toMatchObject({
      type: 'wildcard',
      weight: {
        isSupported: false,
      },
    });
  });

  it('keeps token and wrapper ranges separate for weighted wildcard edits', () => {
    const prompt = '(__lighting/studio__)++, (face)+';
    const occurrence = getPromptWorkbenchOccurrences({
      prompt,
      wildcards,
      wildcardIndexUnavailable: false,
      dynamicPromptMode: 'random',
      supportsAttentionWeights: true,
    })[0];

    expect(occurrence).toMatchObject({ type: 'wildcard' });
    if (occurrence?.type !== 'wildcard') {
      throw new Error('Expected wildcard occurrence');
    }

    expect(replacePromptRange(prompt, occurrence.range, '__@lighting/studio__').prompt).toBe(
      '(__@lighting/studio__)++, (face)+'
    );
    expect(removePromptRange(prompt, occurrence.weight!.range).prompt).toBe('(face)+');
  });
});
