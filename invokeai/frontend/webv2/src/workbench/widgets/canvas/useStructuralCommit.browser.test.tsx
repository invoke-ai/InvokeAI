import type {
  CanvasEditRefusal,
  CanvasLayerPreviewMutation,
  PreparedDocumentEdit,
  StructuralPreviewSession,
} from '@workbench/canvas-engine/api';

import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CanvasEditRefusalNotices } from './CanvasEditRefusalNotices';
import { type CanvasPreviewEngine, type StructuralPreview, useStructuralPreview } from './useStructuralCommit';

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

const createSession = (commitStatus: 'committed' | 'busy' = 'committed') => {
  const session = {
    apply: vi.fn(() => true),
    cancel: vi.fn(),
    commit: vi.fn(() => ({ status: commitStatus })),
  } satisfies StructuralPreviewSession;
  return session;
};

const createEngine = (session: StructuralPreviewSession, commitPrepared = vi.fn(() => ({ status: 'committed' }))) =>
  ({
    document: { model: () => ({}) },
    layers: { beginStructuralPreview: vi.fn(() => session), commitPrepared },
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

  it('commits as a plain prepared edit when another gesture took its session', () => {
    const engine = createEngine(createSession('busy'));
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
