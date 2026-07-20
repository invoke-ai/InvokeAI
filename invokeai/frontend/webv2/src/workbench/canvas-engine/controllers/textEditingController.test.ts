import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { TextEditSession } from '@workbench/canvas-engine/engineStores';

import { DEFAULT_TEXT_OPTIONS } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { TextEditingController } from './textEditingController';

const createHarness = (document: CanvasDocumentContractV2) => {
  let session: TextEditSession | null = null;
  const commitStructural = vi.fn();
  const invalidate = vi.fn();
  const controller = new TextEditingController({
    canEdit: () => true,
    commitStructural,
    createLayerId: () => 'text-new',
    getDocument: () => document,
    invalidate,
    isGestureActive: () => false,
    options: { get: () => DEFAULT_TEXT_OPTIONS },
    session: {
      get: () => session,
      set: (value) => (session = value),
    },
  });
  return { commitStructural, controller, getSession: () => session, invalidate };
};

describe('TextEditingController', () => {
  it('owns create session state and commits one structural layer addition', () => {
    const h = createHarness({ bbox: { height: 1, width: 1, x: 0, y: 0 }, height: 1, layers: [], width: 1 } as never);
    h.controller.openCreate({ x: 10.4, y: 20.6 });
    expect(h.getSession()?.transform).toMatchObject({ x: 10, y: 21 });

    h.controller.commit('hello');
    expect(h.getSession()).toBeNull();
    expect(h.commitStructural).toHaveBeenCalledOnce();
    expect(h.commitStructural.mock.calls[0]?.[0]).toBe('Add text');
  });

  it('uses the registered content reader and cancels empty creates', () => {
    const h = createHarness({ bbox: { height: 1, width: 1, x: 0, y: 0 }, height: 1, layers: [], width: 1 } as never);
    h.controller.openCreate({ x: 0, y: 0 });
    h.controller.setContentReader(() => '   ');

    expect(h.controller.commitOpen()).toBe(true);
    expect(h.getSession()).toBeNull();
    expect(h.commitStructural).not.toHaveBeenCalled();
    expect(h.invalidate).toHaveBeenCalledWith({ overlay: true });
  });
});
