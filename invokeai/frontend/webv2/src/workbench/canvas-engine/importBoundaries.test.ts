import { describe, expect, it } from 'vitest';

const forbiddenPrefixes = [
  '@workbench/backend',
  '@workbench/canvas-operations',
  '@workbench/generation',
  '@workbench/widgets',
];

const modules = import.meta.glob('./**/*.ts', { eager: true, import: 'default', query: '?raw' });

const violations = (): string[] => {
  const result: string[] = [];
  for (const [file, loaded] of Object.entries(modules)) {
    if (file.endsWith('.test.ts') || typeof loaded !== 'string') {
      continue;
    }
    const source = loaded;
    for (const match of source.matchAll(/from\s+['"]([^'"]+)['"]/g)) {
      const specifier = match[1];
      if (specifier && forbiddenPrefixes.some((prefix) => specifier.startsWith(prefix))) {
        result.push(`${file.slice(2)} -> ${specifier}`);
      }
    }
  }
  return result.sort();
};

describe('canvas-engine import boundary', () => {
  it('does not import application workflows, widgets, generation graphs, or backend networking', () => {
    expect(violations()).toEqual([]);
  });
});
