import { describe, expect, it } from 'vitest';

const sources = import.meta.glob(['./react.ts', './ui/GalleryMediaUsagePanel.tsx'], {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>;

const staticSpecifiers = (source: string | undefined): Set<string> =>
  new Set([...(source ?? '').matchAll(/from '([^']+)'/g)].map(([, specifier]) => specifier));

const dynamicSpecifiers = (source: string | undefined): Set<string> =>
  new Set([...(source ?? '').matchAll(/import\('([^']+)'\)/g)].map(([, specifier]) => specifier));

/**
 * The React entry is bundled with the eager gallery-state chunk. A re-export from a lazy gallery module (uploads,
 * the picker) would make the state chunk and the picker chunk import each other, and the picker's module-scope code
 * would run before the state chunk initialised: the production editor failed to start that way once. Dev and test
 * module orders differ, so only the entry's import surface can be checked cheaply.
 */
describe('gallery React entry', () => {
  it('reaches only the eager UI port, the image recovery leaf hook, the media usage panel and core types', () => {
    const source = sources['./react.ts'];

    expect(source).toBeDefined();
    expect(staticSpecifiers(source)).toEqual(
      new Set(['./ui/GalleryUiContext', './ui/useGalleryImageRecovery', './ui/GalleryMediaUsagePanel', './core/types'])
    );
  });

  // The panel rides in the eager chunk so the deletion dialogs' live region exists as they open. It stays a leaf:
  // its report view (Scrollable, tooltips, the report's copy) loads only through its lazy import.
  it('keeps the media usage panel a leaf whose report view loads lazily', () => {
    const source = sources['./ui/GalleryMediaUsagePanel.tsx'];

    expect(source).toBeDefined();
    expect(staticSpecifiers(source)).toEqual(
      new Set(['@features/gallery/core/mediaUsage', '@chakra-ui/react', 'react', 'react-i18next'])
    );
    expect(dynamicSpecifiers(source)).toEqual(new Set(['./GalleryMediaUsageSection']));
    expect(source).toMatch(/^import type \{[^}]*\} from '@features\/gallery\/core\/mediaUsage';$/m);
  });
});
