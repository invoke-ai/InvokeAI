import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { useModelGridSize } from './bboxGrid';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** Stands in for the canvas widget's `engine.viewport.setBboxGrid` feed and the header's fit-bbox. */
const GridProbe = ({ base }: { base: string }) => <span data-testid="grid">{useModelGridSize(base)}</span>;

const render = async (base: string): Promise<void> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(<GridProbe base={base} />);
  });
};

const grid = (): string | null | undefined => host?.querySelector('[data-testid="grid"]')?.textContent;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  resetArchitectureCapabilities();
});

describe('useModelGridSize', () => {
  it('re-reads the grid when the capability table lands after mount', async () => {
    // Capability arrival must update a restored unchanged base's snapping grid rather than retaining boot fallback
    // 8.
    await render('wan');
    expect(grid()).toBe('8');

    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    expect(grid()).toBe('16');
  });

  it('drops back to the default when the table is dropped on an account change', async () => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
    await render('cogview4');
    expect(grid()).toBe('32');

    await act(() => {
      resetArchitectureCapabilities();
    });

    expect(grid()).toBe('8');
  });
});
