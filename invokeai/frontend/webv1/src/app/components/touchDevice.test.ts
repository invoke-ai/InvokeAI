import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

const css = readFileSync(new URL('./touchDevice.css', import.meta.url), 'utf8');

describe('touchDevice.css', () => {
  it('hides tooltips only after touch input has been detected', () => {
    expect(css).toMatch(/\.invokeai-touch-device\s+\[role='tooltip'\]\s*{/);
  });

  it('does not force all tooltips invisible', () => {
    expect(css).not.toMatch(/@media\s*\([^)]*hover[^)]*\)/);
  });
});
