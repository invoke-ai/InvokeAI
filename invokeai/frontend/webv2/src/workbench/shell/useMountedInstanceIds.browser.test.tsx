import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { useMountedInstanceIds } from './useMountedInstanceIds';

// `@testing-library/react` is not a dependency of this package, so the hook is
// driven through a probe component the way the other browser suites here do.
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/**
 * The value the probe last produced. Module scope keeps the probe prop-free, and
 * it is published from an effect so the probe stays render-pure.
 */
const latest: { ids: string[] } = { ids: [] };

const Probe = ({ activeId, limit }: { activeId: string | undefined; limit?: number }) => {
  const ids = useMountedInstanceIds(activeId, limit);

  useEffect(() => {
    latest.ids = ids;
  }, [ids]);

  return null;
};

const renderHook = async (activeId: string | undefined, limit?: number) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  const rerender = async (nextActiveId: string | undefined) => {
    await act(async () => {
      root?.render(<Probe activeId={nextActiveId} limit={limit} />);
      await Promise.resolve();
    });
  };

  await rerender(activeId);

  return { rerender };
};

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
  latest.ids = [];
});

describe('mounted instance ids', () => {
  it('remembers instances that have been shown', async () => {
    const { rerender } = await renderHook('canvas');

    expect(latest.ids).toEqual(['canvas']);

    await rerender('workflow:center');
    expect(latest.ids).toEqual(['canvas', 'workflow:center']);

    await rerender('canvas');
    expect(latest.ids).toEqual(['workflow:center', 'canvas']);
  });

  it('evicts least-recently-shown beyond the limit', async () => {
    const { rerender } = await renderHook('a', 2);

    await rerender('b');
    await rerender('c');

    expect(latest.ids).toEqual(['b', 'c']);
  });

  it('keeps the set stable when the active id does not change', async () => {
    const { rerender } = await renderHook('canvas');
    const first = latest.ids;

    await rerender('canvas');

    expect(latest.ids).toBe(first);
  });

  it('tolerates no active instance', async () => {
    await renderHook(undefined);

    expect(latest.ids).toEqual([]);
  });

  it('starts remembering once an instance finally arrives', async () => {
    const { rerender } = await renderHook(undefined);

    await rerender('canvas');

    expect(latest.ids).toEqual(['canvas']);
  });
});
