import { describe, expect, it } from 'vitest';

const modules = import.meta.glob('../widgets/**/*.{ts,tsx}', { eager: true, import: 'default', query: '?raw' });

const fullEngineImports = (): string[] => {
  const result: string[] = [];
  const pattern =
    /import(?:\s+type)?\s*\{[^}]*\bCanvasEngine\b[^}]*\}\s*from\s*['"]@workbench\/canvas-engine\/engine['"]/s;
  for (const [file, source] of Object.entries(modules)) {
    if (file.endsWith('.test.ts') || file.endsWith('.test.tsx') || typeof source !== 'string') {
      continue;
    }
    if (pattern.test(source)) {
      result.push(file.replace('../widgets/', ''));
    }
  }
  return result.sort();
};

describe('full CanvasEngine consumer boundary', () => {
  it('keeps application composition consumers off the core CanvasEngine type', () => {
    expect(fullEngineImports()).toEqual([]);
  });
});
