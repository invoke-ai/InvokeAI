import { describe, expect, it, vi } from 'vitest';

const toSvg = vi.hoisted(() => vi.fn());
vi.mock('html-to-image', () => ({ toSvg }));

import { rasterizeWorkflowImage } from './workflowImageRaster';

const containsAbortSignal = (value: unknown): boolean =>
  value instanceof AbortSignal ||
  (typeof value === 'object' && value !== null && Object.values(value).some(containsAbortSignal));

describe('workflow image rasterizer', () => {
  it('serializes with fonts but without the abort signal, then skips every later stage once aborted', async () => {
    const controller = new AbortController();
    const element = {} as HTMLElement;
    toSvg.mockImplementation(() => {
      controller.abort();
      return Promise.resolve('data:image/svg+xml,');
    });
    const createElement = vi.fn();
    vi.stubGlobal('document', { createElement });

    try {
      await expect(
        rasterizeWorkflowImage(element, {
          backgroundColor: 'rgb(1, 2, 3)',
          height: 200,
          signal: controller.signal,
          styleProperties: ['color'],
          width: 300,
        })
      ).rejects.toMatchObject({ name: 'AbortError' });

      expect(toSvg).toHaveBeenCalledWith(
        element,
        expect.objectContaining({
          height: 200,
          includeStyleProperties: ['color'],
          skipFonts: false,
          width: 300,
        })
      );
      // html-to-image caches a failed fetch's result, so an aborted font fetch would poison every later capture.
      expect(containsAbortSignal(toSvg.mock.calls[0]![1])).toBe(false);
      expect(createElement).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
