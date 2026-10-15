import type {
  CanvasEditRefusal,
  CanvasLayerPreviewMutation,
  PreparedDocumentEdit,
  StructuralPreviewSession,
} from '@workbench/canvas-engine/api';

import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CanvasEditRefusalNotices } from './CanvasEditRefusalNotices';
import {
  baselinePatch,
  type CanvasPreviewEngine,
  type StructuralPreview,
  useStructuralPreview,
} from './useStructuralCommit';

const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const cleanup of cleanups.splice(0)) {
    cleanup();
  }
  vi.clearAllMocks();
});

const render = (element: React.ReactElement): void => {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(element));
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
};

const opacity = (value: number): CanvasLayerPreviewMutation => ({
  id: 'layer',
  patch: { opacity: value },
  type: 'updateCanvasLayer',
});
const EDIT = { forward: opacity(0.5), inverse: opacity(1) } as unknown as PreparedDocumentEdit;

/** A session mock; `active: false` models one the engine ended (a replay or a commit from elsewhere landed). */
const createSession = (commitStatus: 'committed' | 'busy' = 'committed', active = true) => {
  const session = {
    apply: vi.fn(() => active),
    baseline: vi.fn(() => (active ? opacity(1) : null)),
    cancel: vi.fn(),
    commit: vi.fn(() => ({ status: commitStatus })),
    isActive: vi.fn(() => active),
  } satisfies StructuralPreviewSession;
  return session;
};

const createEngine = (session: StructuralPreviewSession, commitPrepared = vi.fn(() => ({ status: 'committed' }))) =>
  ({
    document: { model: () => ({}) },
    layers: { beginStructuralPreview: vi.fn(() => session), commitPrepared, endStructuralPreview: vi.fn() },
  }) as unknown as CanvasPreviewEngine & { layers: { commitPrepared: typeof commitPrepared } };

const PreviewProbe = ({ engine, ref }: { engine: CanvasPreviewEngine; ref: Ref<StructuralPreview> }) => {
  const preview = useStructuralPreview(engine);
  useImperativeHandle(ref, () => preview, [preview]);
  return null;
};

const renderPreview = (engine: CanvasPreviewEngine): StructuralPreview => {
  const ref = createRef<StructuralPreview>();
  render(<PreviewProbe ref={ref} engine={engine} />);
  return ref.current!;
};

/** An engine whose only capability is the refusal channel, and the emitter that drives it. */
const createRefusalEngine = () => {
  const listeners: ((refusal: CanvasEditRefusal) => void)[] = [];
  return {
    emit: (refusal: CanvasEditRefusal) => listeners.forEach((listener) => listener(refusal)),
    engine: {
      tools: {
        onEditRefused: (listener: (refusal: CanvasEditRefusal) => void) => {
          listeners.push(listener);
          return () => undefined;
        },
      },
    } as never,
  };
};

describe('useStructuralPreview', () => {
  it('records a previewed gesture through the session that previewed it', () => {
    const session = createSession();
    const engine = createEngine(session);
    const preview = renderPreview(engine);

    expect(preview.preview(opacity(0.7))).toBe(true);
    expect(preview.preview(opacity(0.5))).toBe(true);
    const outcome = preview.commit('Opacity', () => ({ edit: EDIT, status: 'prepared' }) as never);

    expect(engine.layers.beginStructuralPreview).toHaveBeenCalledOnce();
    expect(session.apply).toHaveBeenCalledTimes(2);
    expect(session.commit).toHaveBeenCalledWith('Opacity', EDIT);
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
    expect(outcome).toEqual({ status: 'committed' });
  });

  it('commits as a plain prepared edit when the engine ended its session', () => {
    const engine = createEngine(createSession('busy', false));
    const preview = renderPreview(engine);
    preview.preview(opacity(0.5));

    preview.commit('Opacity', () => ({ edit: EDIT, status: 'prepared' }) as never);
    expect(engine.layers.commitPrepared).toHaveBeenCalledWith('Opacity', EDIT);
  });

  it('cancels its session and reports when the edit cannot be prepared', () => {
    const session = createSession();
    const preview = renderPreview(createEngine(session));
    preview.preview(opacity(0.5));

    preview.commit('Opacity', () => ({ status: 'locked' }) as never);
    expect(session.cancel).toHaveBeenCalledOnce();
    expect(notify.error).toHaveBeenCalledWith(
      'widgets.canvas.structural.failed',
      'widgets.canvas.structural.refusedLocked'
    );
  });

  it("prepares from the session's baseline, and from the live document once the engine ended the session", () => {
    const live = createSession();
    const prepare = vi.fn(() => ({ edit: EDIT, status: 'prepared' }) as never);
    const livePreview = renderPreview(createEngine(live));
    livePreview.preview(opacity(0.5));
    livePreview.commit('Opacity', prepare);
    expect(prepare).toHaveBeenLastCalledWith(expect.anything(), opacity(1));

    const ended = createSession('busy', false);
    const engine = createEngine(ended);
    const preview = renderPreview(engine);
    preview.preview(opacity(0.5));
    preview.commit('Opacity', prepare);

    expect(prepare).toHaveBeenLastCalledWith(expect.anything(), null);
    expect(engine.layers.commitPrepared).toHaveBeenCalledWith('Opacity', EDIT);
  });

  it('stays silent when the document change that ended the gesture also refuses its edit, and reports it otherwise', () => {
    const ended = createSession('busy', false);
    const endedPreview = renderPreview(createEngine(ended));
    endedPreview.preview(opacity(0.5));

    expect(endedPreview.commit('Opacity', () => ({ ids: ['layer'], status: 'missing' }) as never)).toEqual({
      refusal: { ids: ['layer'], status: 'missing' },
      status: 'refused',
    });
    // An undone conversion or unlock refuses the late commit too; that refusal is the undo's, not the user's.
    endedPreview.preview(opacity(0.5));
    expect(
      endedPreview.commit('Opacity', () => ({ actual: 'raster', expected: ['control'], status: 'wrong-type' }) as never)
    ).toMatchObject({
      status: 'refused',
    });
    expect(notify.error).not.toHaveBeenCalled();

    const live = createSession();
    const livePreview = renderPreview(createEngine(live));
    livePreview.preview(opacity(0.5));
    livePreview.commit('Opacity', () => ({ ids: ['layer'], status: 'missing' }) as never);

    expect(live.cancel).toHaveBeenCalledOnce();
    expect(notify.error).toHaveBeenCalledWith(
      'widgets.canvas.structural.failed',
      'widgets.canvas.structural.refusedMissing'
    );
  });
});

describe('useStructuralPreview over the engine', () => {
  it('keeps a gesture whose later previews were refused, so its release restores the baseline rather than recording the previewed value', () => {
    const locked = { value: false };
    const stub = createStructuralEngineStub({
      locked,
      schedulePreview: (flush) => {
        flush();
        return () => undefined;
      },
    });
    const preview = renderPreview(stub.engine as unknown as CanvasPreviewEngine);
    const opacityOf = () => getDocumentLayer(stub.document(), 'layer')?.opacity;

    expect(preview.preview(opacity(0.5))).toBe(true);
    expect(opacityOf()).toBe(0.5);
    locked.value = true;
    expect(preview.preview(opacity(0.3))).toBe(false);
    expect(preview.baseline()).toEqual(opacity(1));

    const outcome = preview.commit('Opacity', (model, baseline) =>
      model.prepare({
        before: baselinePatch(baseline, { opacity: 0.3 }),
        id: 'layer',
        patch: { opacity: 0.3 },
        type: 'patch',
      })
    );

    expect(outcome).toEqual({ status: 'busy' });
    expect(opacityOf()).toBe(1);
    expect(stub.history.canUndo()).toBe(false);
    expect(preview.baseline()).toBeNull();
    expect(notify.error).not.toHaveBeenCalled();
  });
});

describe('useStructuralPreview with a gesture superseded by another', () => {
  it('commits the superseded gesture from the committed document, ending the newer preview first', () => {
    const stub = createStructuralEngineStub({
      schedulePreview: (flush) => {
        flush();
        return () => undefined;
      },
    });
    const engine = stub.engine as unknown as CanvasPreviewEngine;
    const tint = renderPreview(engine);
    const scrub = renderPreview(engine);
    const layer = () => getDocumentLayer(stub.document(), 'layer');

    expect(tint.preview({ id: 'layer', patch: { name: 'Draft' }, type: 'updateCanvasLayer' })).toBe(true);
    // A newer gesture's session restores the first one's baseline and holds its own preview.
    expect(scrub.preview(opacity(0.3))).toBe(true);
    expect(layer()).toMatchObject({ name: 'Layer', opacity: 0.3 });

    const outcome = tint.commit('Name', (model, baseline) =>
      model.prepare({
        before: baselinePatch(baseline, { name: 'Tinted' }),
        id: 'layer',
        patch: { name: 'Tinted' },
        type: 'patch',
      })
    );

    expect(outcome).toEqual({ status: 'committed' });
    expect(layer()).toMatchObject({ name: 'Tinted', opacity: 1 });
    expect(stub.history.entries().past).toEqual(['Name']);
    expect(notify.error).not.toHaveBeenCalled();
  });
});

describe('CanvasEditRefusalNotices', () => {
  it.each<[CanvasEditRefusal, string]>([
    ['over-budget', 'widgets.canvas.structural.overBudget'],
    ['busy', 'widgets.canvas.structural.busy'],
  ])('explains a %s refusal of an edit nobody else reports', (refusal, message) => {
    const { emit, engine } = createRefusalEngine();
    render(<CanvasEditRefusalNotices engine={engine} />);

    emit(refusal);
    expect(notify.error).toHaveBeenCalledWith('widgets.canvas.structural.failed', message);
  });
});
