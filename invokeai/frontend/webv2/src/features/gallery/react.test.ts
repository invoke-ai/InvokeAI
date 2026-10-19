import { describe, expect, it } from 'vitest';

const sources = import.meta.glob('./react.ts', { eager: true, import: 'default', query: '?raw' }) as Record<
  string,
  string
>;

/**
 * The React entry is bundled with the eager gallery-state chunk. A re-export from a lazy gallery module (uploads,
 * the picker) would make the state chunk and the picker chunk import each other, and the picker's module-scope code
 * would run before the state chunk initialised: the production editor failed to start that way once. Dev and test
 * module orders differ, so only the entry's import surface can be checked cheaply.
 */
describe('gallery React entry', () => {
  it('reaches only the eager UI port and core types', () => {
    const source = sources['./react.ts'];
    const specifiers = [...(source ?? '').matchAll(/from '([^']+)'/g)].map(([, specifier]) => specifier);

    expect(source).toBeDefined();
    expect(new Set(specifiers)).toEqual(new Set(['./ui/GalleryUiContext', './core/types']));
  });
});
