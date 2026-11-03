import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';

import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it } from 'vitest';

import { useGallerySemanticImageQuery } from './GalleryWidgetView';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const FIRST_A = { kind: 'url', url: 'https://a.test/image.png' } as const;
const SECOND_A = { kind: 'url', url: 'https://a.test/other.png' } as const;
const ONLY_B = { imageName: 'b.png', kind: 'image' } as const;

it('keeps semantic query identity stable per gallery consumer', async () => {
  const references: Record<'a' | 'b', (GallerySemanticReference | null)[]> = { a: [], b: [] };
  const host = document.createElement('div');
  const root = createRoot(host);

  document.body.append(host);

  const Probe = ({ id, value }: { id: 'a' | 'b'; value: unknown }) => {
    references[id].push(useGallerySemanticImageQuery(value));

    return null;
  };

  await act(() =>
    root.render(
      <>
        <Probe id="a" value={FIRST_A} />
        <Probe id="b" value={ONLY_B} />
      </>
    )
  );
  await act(() =>
    root.render(
      <>
        <Probe id="a" value={FIRST_A} />
        <Probe id="b" value={ONLY_B} />
      </>
    )
  );
  await act(() =>
    root.render(
      <>
        <Probe id="a" value={SECOND_A} />
        <Probe id="b" value={ONLY_B} />
      </>
    )
  );

  expect(references.a[1]).toBe(references.a[0]);
  expect(references.b[1]).toBe(references.b[0]);
  expect(references.a[2]).not.toBe(references.a[1]);
  expect(references.b[2]).toBe(references.b[1]);

  await act(() => root.unmount());
  host.remove();
});
