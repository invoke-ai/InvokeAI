import { beforeEach, describe, expect, it, vi } from 'vitest';

import { getWidgetReadyMark, LAUNCHPAD_READY_MARK, markSemanticReady } from './semanticReady';

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('semanticReady', () => {
  it('uses stable route and widget milestone names', () => {
    expect(LAUNCHPAD_READY_MARK).toBe('invokeai:ready:launchpad');
    expect(getWidgetReadyMark('center', 'canvas')).toBe('invokeai:ready:widget:center:canvas');
  });

  it('replaces an older mark before recording the current milestone', () => {
    const clear = vi.spyOn(performance, 'clearMarks').mockImplementation(() => undefined);
    const mark = vi.spyOn(performance, 'mark').mockImplementation((name: string) => ({ name }) as PerformanceMark);

    markSemanticReady(LAUNCHPAD_READY_MARK);

    expect(clear).toHaveBeenCalledWith(LAUNCHPAD_READY_MARK);
    expect(mark).toHaveBeenCalledWith(LAUNCHPAD_READY_MARK);
    expect(clear.mock.invocationCallOrder[0]).toBeLessThan(mark.mock.invocationCallOrder[0]);
  });
});
