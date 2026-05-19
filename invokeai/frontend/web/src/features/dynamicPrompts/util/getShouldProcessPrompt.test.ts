import { describe, expect, it } from 'vitest';

import { getShouldProcessPrompt } from './getShouldProcessPrompt';

describe('getShouldProcessPrompt', () => {
  it('detects variant syntax', () => {
    expect(getShouldProcessPrompt('a {red|blue} car')).toBe(true);
  });

  it('detects wildcard syntax', () => {
    expect(getShouldProcessPrompt('a __color__ car')).toBe(true);
  });

  it('ignores plain prompts', () => {
    expect(getShouldProcessPrompt('a red car')).toBe(false);
  });
});
