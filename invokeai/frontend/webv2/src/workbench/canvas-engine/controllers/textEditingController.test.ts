import type { StructuralCommitResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { TextEditSession, TextToolOptions } from '@workbench/canvas-engine/engineStores';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { EMPTY_STACKS } from '@workbench/canvas-engine/document/documentTree';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { DEFAULT_TEXT_OPTIONS } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { TextEditingController } from './textEditingController';

const createHarness = (document: CanvasDocumentContractV3, textOptions: TextToolOptions = DEFAULT_TEXT_OPTIONS) => {
  let session: TextEditSession | null = null;
  const commitStructural = vi.fn<(label: string, forward: unknown, inverse: unknown) => StructuralCommitResult>(() => ({
    status: 'committed',
  }));
  const invalidate = vi.fn();
  const controller = new TextEditingController({
    canEdit: () => true,
    captureInsertionAnchor: createTestInsertionAnchorCapture('p', () => document?.stacks ?? EMPTY_STACKS),
    colors: { get: () => ({ background: '#ffffff', foreground: '#123456' }) },
    commitStructural,
    createLayerId: () => 'text-new',
    getDocument: () => document,
    invalidate,
    isGestureActive: () => false,
    options: { get: () => textOptions },
    session: {
      get: () => session,
      set: (value) => (session = value),
    },
  });
  return { commitStructural, controller, getSession: () => session, invalidate };
};

describe('TextEditingController', () => {
  it('owns create session state and commits one structural layer addition', () => {
    const h = createHarness({
      bbox: { height: 1, width: 1, x: 0, y: 0 },
      height: 1,
      stacks: stacksFrom([]),
      width: 1,
    } as never);
    h.controller.openCreate({ x: 10.4, y: 20.6 });
    expect(h.getSession()?.transform).toMatchObject({ x: 10, y: 21 });

    h.controller.commit('hello');
    expect(h.getSession()).toBeNull();
    expect(h.commitStructural).toHaveBeenCalledOnce();
    expect(h.commitStructural.mock.calls[0]?.[0]).toBe('Add text');
    // The session's color is the active foreground at open, not a tool option.
    const forward = h.commitStructural.mock.calls[0]?.[1] as { layer?: { source?: { color?: string } } };
    expect(forward.layer?.source?.color).toBe('#123456');
  });

  it('uses the registered content reader and cancels empty creates', () => {
    const h = createHarness({
      bbox: { height: 1, width: 1, x: 0, y: 0 },
      height: 1,
      stacks: stacksFrom([]),
      width: 1,
    } as never);
    h.controller.openCreate({ x: 0, y: 0 });
    h.controller.setContentReader(() => '   ');

    expect(h.controller.commitOpen()).toBe(true);
    expect(h.getSession()).toBeNull();
    expect(h.commitStructural).not.toHaveBeenCalled();
    expect(h.invalidate).toHaveBeenCalledWith({ overlay: true });
  });

  it('carries custom font identity, style, and exact variation coordinates into the structural source', () => {
    const options: TextToolOptions = {
      ...DEFAULT_TEXT_OPTIONS,
      fontFamily: 'Catalog Family',
      fontRef: { contentHash: 'hash-1', family: 'Catalog Family', id: 'font-1', label: 'Catalog Regular' },
      fontStyle: 'italic',
      fontVariations: { opsz: 14, wght: 650 },
      fontWeight: 650,
    };
    const h = createHarness(
      {
        bbox: { height: 1, width: 1, x: 0, y: 0 },
        height: 1,
        stacks: stacksFrom([]),
        width: 1,
      } as never,
      options
    );

    h.controller.openCreate({ x: 0, y: 0 });
    h.controller.commit('hello');

    const forward = h.commitStructural.mock.calls[0]?.[1] as {
      layer?: { source?: { fontRef?: unknown; fontStyle?: string; fontVariations?: unknown } };
    };
    expect(forward.layer?.source).toMatchObject({
      fontRef: options.fontRef,
      fontStyle: 'italic',
      fontVariations: { opsz: 14, wght: 650 },
    });
  });
});

describe('TextEditingController refusals', () => {
  it('keeps the session for a transient refusal and drops it when the target is gone', () => {
    const h = createHarness({
      bbox: { height: 1, width: 1, x: 0, y: 0 },
      height: 1,
      stacks: stacksFrom([]),
      width: 1,
    } as never);
    h.commitStructural.mockReturnValueOnce({ status: 'busy' as const });
    h.controller.openCreate({ x: 0, y: 0 });

    expect(h.controller.commit('hello')).toEqual({ status: 'busy' });
    expect(h.getSession()).not.toBeNull();

    h.commitStructural.mockReturnValueOnce({ status: 'dispatch-rejected' as const });
    expect(h.controller.commit('hello')).toEqual({ status: 'dispatch-rejected' });
    expect(h.getSession()).toBeNull();
  });

  it('reports nothing to commit for an empty creation', () => {
    const h = createHarness({
      bbox: { height: 1, width: 1, x: 0, y: 0 },
      height: 1,
      stacks: stacksFrom([]),
      width: 1,
    } as never);
    h.controller.openCreate({ x: 0, y: 0 });

    expect(h.controller.commit('   ')).toBeNull();
    expect(h.getSession()).toBeNull();
    expect(h.commitStructural).not.toHaveBeenCalled();
  });
});
