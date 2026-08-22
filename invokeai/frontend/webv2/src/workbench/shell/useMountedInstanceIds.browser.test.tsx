import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import {
  getActiveInstanceIdsOutside,
  useMountedInstanceIds,
  withoutInstancesShownElsewhere,
} from './useMountedInstanceIds';

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

const Probe = ({ activeId, limit, resetKey }: { activeId: string | undefined; limit?: number; resetKey: string }) => {
  const ids = useMountedInstanceIds(activeId, resetKey, limit);

  useEffect(() => {
    latest.ids = ids;
  }, [ids]);

  return null;
};

const renderHook = async (activeId: string | undefined, limit?: number) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  const rerender = async (nextActiveId: string | undefined, resetKey = 'project-a') => {
    await act(async () => {
      root?.render(<Probe activeId={nextActiveId} limit={limit} resetKey={resetKey} />);
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

  it('forgets everything the previous project had shown', async () => {
    const { rerender } = await renderHook('canvas');

    await rerender('workflow:center');
    expect(latest.ids).toEqual(['canvas', 'workflow:center']);

    // Instance ids are identical in every project, so a remembered id would
    // otherwise resolve to a real — but wrong — instance of the new project.
    await rerender('workflow:center', 'project-b');
    expect(latest.ids).toEqual(['workflow:center']);
  });
});

describe('instances shown elsewhere', () => {
  it('drops a kept instance another region is actively rendering', () => {
    expect(withoutInstancesShownElsewhere(['preview', 'canvas'], 'canvas', ['preview'])).toEqual(['canvas']);
  });

  it('never drops the active instance of the region asking', () => {
    expect(withoutInstancesShownElsewhere(['gallery', 'preview'], 'preview', ['preview'])).toEqual([
      'gallery',
      'preview',
    ]);
  });

  it('keeps the list identity when nothing is shown elsewhere', () => {
    const mountedIds = ['preview', 'canvas'];

    expect(withoutInstancesShownElsewhere(mountedIds, 'canvas', ['layers'])).toBe(mountedIds);
  });

  it('reads actives only, never region membership', () => {
    const widgetRegions = {
      bottom: { activeInstanceId: 'gallery:bottom', instanceIds: ['gallery:bottom'] },
      center: { activeInstanceId: 'canvas', instanceIds: ['canvas', 'preview'] },
      left: { activeInstanceId: 'generate', instanceIds: ['generate', 'upscale'] },
      right: { activeInstanceId: 'layers', instanceIds: ['layers', 'preview', 'gallery'] },
    };

    // `preview` is a member of the right rail but not its active instance, so it
    // stays available for the centre to keep.
    expect(getActiveInstanceIdsOutside(widgetRegions, 'center')).toEqual(['gallery:bottom', 'generate', 'layers']);
  });

  it('includes floating instance ids, since a float removes an instance from its region entirely', () => {
    const widgetRegions = {
      center: { activeInstanceId: 'canvas', instanceIds: ['canvas'] },
      right: { activeInstanceId: 'layers', instanceIds: ['layers'] },
    };

    // `floatWidget` hands the region off to a fallback, so `image-map` no
    // longer appears as any region's active — only the floating map catches
    // it, which is what keeps the region's kept copy from shadowing the
    // floating window.
    expect(getActiveInstanceIdsOutside(widgetRegions, 'right', { 'image-map': {} })).toEqual(['canvas', 'image-map']);
  });
});
