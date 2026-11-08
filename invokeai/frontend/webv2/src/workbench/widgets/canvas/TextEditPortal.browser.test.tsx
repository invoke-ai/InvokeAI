import type { TextEditSession } from '@workbench/canvas-engine/api';

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { TextEditPortal } from './TextEditPortal';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounted: { root: Root; host: HTMLDivElement }[] = [];

afterEach(() => {
  while (mounted.length > 0) {
    const entry = mounted.pop();
    if (entry) {
      act(() => entry.root.unmount());
      entry.host.remove();
    }
  }
});

const session = (overrides: Partial<TextEditSession> = {}): TextEditSession =>
  ({
    id: 1,
    layerId: 'layer-1',
    mode: 'edit',
    startSource: null,
    source: {
      align: 'left',
      color: '#ff0000',
      content: 'hello',
      fontFamily: 'Inter',
      fontSize: 24,
      fontWeight: 400,
      lineHeight: 1.2,
    },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
    ...overrides,
  }) as unknown as TextEditSession;

const createEngine = (
  initialSession: TextEditSession | null,
  fonts?: {
    ensurePreview: (source: TextEditSession['source'], signal?: AbortSignal) => Promise<string>;
    resolveFamily: (source: TextEditSession['source']) => string;
    subscribe: (listener: () => void) => () => void;
  },
  closeOnCommand = false
) => {
  let current = initialSession;
  const listeners = new Set<() => void>();
  const layers = {
    cancelTextEdit: vi.fn(),
    commitTextEdit: vi.fn(),
    setTextEditContentReader: vi.fn(),
  };
  const viewport = {
    documentToScreen: ({ x, y }: { x: number; y: number }) => ({ x, y }),
    getState: () => ({ pan: { x: 0, y: 0 }, zoom: 1 }),
    getZoom: () => 1,
    subscribe: () => () => {},
  };
  const engine = {
    interaction: {
      get: (key: string) => (key === 'textEditSession' ? current : undefined),
      subscribe: (_key: string, listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    layers,
    viewport: { getViewport: () => viewport },
    ...(fonts ? { fonts } : {}),
  };
  const setSession = (next: TextEditSession | null) => {
    current = next;
    listeners.forEach((listener) => listener());
  };
  if (closeOnCommand) {
    layers.cancelTextEdit.mockImplementation(() => setSession(null));
    layers.commitTextEdit.mockImplementation(() => {
      setSession(null);
      return null;
    });
  }
  return { engine: engine as unknown as Parameters<typeof TextEditPortal>[0]['engine'], layers, setSession };
};

const mount = (engine: Parameters<typeof TextEditPortal>[0]['engine']) => {
  const host = window.document.createElement('div');
  host.tabIndex = -1;
  window.document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<TextEditPortal engine={engine} />));
  mounted.push({ host, root });
  return { host, root };
};

const editableIn = (host: HTMLElement): HTMLElement => {
  const el = host.querySelector<HTMLElement>('[role="textbox"]');
  if (!el) {
    throw new Error('editable not rendered');
  }
  return el;
};

describe('TextEditPortal', () => {
  it('renders nothing without an active session', () => {
    const { engine } = createEngine(null);
    const { host } = mount(engine);
    expect(host.querySelector('[role="textbox"]')).toBeNull();
  });

  it('seeds the session content and focuses the editable exactly once', () => {
    const { engine } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    expect(el.textContent).toBe('hello');
    expect(window.document.activeElement).toBe(el);
    expect(el.dataset.seeded).toBe('true');
    expect(el.getAttribute('aria-label')).toBe('widgets.canvas.toolOptions.textEdit');
    expect(el.getAttribute('aria-multiline')).toBe('true');
  });

  it('places the caret at the end of the seeded content', () => {
    const { engine } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    const selection = window.getSelection();
    expect(selection?.isCollapsed).toBe(true);
    expect(el.contains(selection?.anchorNode ?? null)).toBe(true);
    expect(selection?.anchorOffset).toBe(el.childNodes.length);
  });

  it('registers a live content reader and clears it on unmount', () => {
    const { engine, layers } = createEngine(session());
    const { host, root } = mount(engine);
    const el = editableIn(host);
    const reader = layers.setTextEditContentReader.mock.calls.at(-1)?.[0] as () => string;
    expect(typeof reader).toBe('function');

    el.textContent = 'typed live';
    expect(reader()).toBe('typed live');

    act(() => root.unmount());
    mounted.pop();
    host.remove();
    expect(layers.setTextEditContentReader).toHaveBeenLastCalledWith(null);
  });

  it('commits the current text on blur', () => {
    const { engine, layers } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    el.textContent = 'edited';
    // React maps `onBlur` to the bubbling `focusout`, so blur the element for real.
    act(() => el.blur());
    expect(layers.commitTextEdit).toHaveBeenCalledWith('edited');
  });

  it.each(['busy', 'gesture-active', 'not-ready'] as const)(
    'keeps focus in the editable when a keyboard commit is refused as %s',
    (status) => {
      const { engine, layers } = createEngine(session());
      layers.commitTextEdit.mockReturnValue({ status });
      const { host } = mount(engine);
      const el = editableIn(host);
      el.textContent = 'retry this text';
      act(() => {
        el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter', metaKey: true }));
      });
      expect(layers.commitTextEdit).toHaveBeenCalledExactlyOnceWith('retry this text');
      expect(host.querySelector('[role="textbox"]')).toBe(el);
      expect(document.activeElement).toBe(el);
    }
  );

  it.each(['commit', 'cancel'] as const)(
    'restores CanvasSurface focus when keyboard %s unmounts the editable',
    (action) => {
      const { engine, layers } = createEngine(session(), undefined, true);
      const { host } = mount(engine);
      const el = editableIn(host);

      act(() => {
        el.dispatchEvent(
          new KeyboardEvent('keydown', {
            bubbles: true,
            key: action === 'commit' ? 'Enter' : 'Escape',
            ...(action === 'commit' ? { ctrlKey: true } : {}),
          })
        );
      });

      expect(action === 'commit' ? layers.commitTextEdit : layers.cancelTextEdit).toHaveBeenCalledTimes(1);
      expect(layers.commitTextEdit).toHaveBeenCalledTimes(action === 'commit' ? 1 : 0);
      expect(host.querySelector('[role="textbox"]')).toBeNull();
      expect(document.activeElement).toBe(host);
    }
  );

  it('keeps the pointer blur destination focused while committing the edited text', () => {
    const { engine, layers } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    const destination = document.createElement('button');
    destination.textContent = 'Elsewhere';
    host.append(destination);
    el.textContent = 'pointer edit';

    act(() => destination.focus());
    expect(layers.commitTextEdit).toHaveBeenCalledWith('pointer edit');
    expect(document.activeElement).toBe(destination);
  });

  it('commits on mod+enter and cancels on escape', () => {
    const { engine, layers } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    el.textContent = 'committed';

    act(() => {
      el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter', metaKey: true }));
    });
    expect(layers.commitTextEdit).toHaveBeenCalledWith('committed');

    act(() => {
      el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' }));
    });
    expect(layers.cancelTextEdit).toHaveBeenCalled();
  });

  it('plain enter inserts a newline rather than committing', () => {
    const { engine, layers } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    act(() => el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' })));
    expect(layers.commitTextEdit).not.toHaveBeenCalled();
    expect(layers.cancelTextEdit).not.toHaveBeenCalled();
  });

  it('stops every keystroke from escaping to canvas hotkeys', () => {
    const { engine } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    const onWindowKeyDown = vi.fn();
    window.addEventListener('keydown', onWindowKeyDown);
    try {
      for (const key of ['b', 'Escape', 'Delete', 'Enter']) {
        act(() => el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key })));
      }
      expect(onWindowKeyDown).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('keydown', onWindowKeyDown);
    }
  });

  it('remounts and re-seeds when the session id changes', () => {
    const { engine, setSession } = createEngine(session());
    const { host } = mount(engine);
    expect(editableIn(host).textContent).toBe('hello');

    act(() => {
      setSession(session({ id: 2, source: { ...session().source, content: 'second' } }));
    });
    expect(editableIn(host).textContent).toBe('second');
  });

  it('positions the editable from the session transform and document styles', () => {
    const { engine } = createEngine(session());
    const { host } = mount(engine);
    const el = editableIn(host);
    expect(el.style.transform).toContain('translate(10px, 20px)');
    expect(el.style.fontSize).toBe('24px');
    expect(el.style.whiteSpace).toBe('pre');
    expect(el.style.color).toBe('rgb(255, 0, 0)');
  });

  it('loads the active custom source and updates overlay metrics when its family becomes ready', async () => {
    let resolvePreview!: (family: string) => void;
    let resolvedFamily = 'fallback-family';
    const listeners = new Set<() => void>();
    const ensurePreview = vi.fn(
      (_source: TextEditSession['source'], _signal?: AbortSignal) =>
        new Promise<string>((resolve) => {
          resolvePreview = resolve;
        })
    );
    const fonts = {
      ensurePreview,
      resolveFamily: vi.fn(() => resolvedFamily),
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };
    const source = {
      ...session().source,
      fontFamily: 'Catalog Family',
      fontRef: { contentHash: 'hash', family: 'Catalog Family', id: 'font-1', label: 'Catalog Regular' },
    };
    const { engine } = createEngine(session({ source }), fonts);
    const { host, root } = mount(engine);
    expect(ensurePreview).toHaveBeenCalledWith(source, expect.any(AbortSignal));
    expect(editableIn(host).style.fontFamily).toBe('fallback-family');
    expect(listeners.size).toBeGreaterThan(0);

    await act(async () => {
      resolvedFamily = 'loaded-family';
      resolvePreview('loaded-family');
      listeners.forEach((listener) => listener());
      await Promise.resolve();
    });
    expect(fonts.resolveFamily).toHaveBeenCalledWith(source);
    expect(fonts.resolveFamily.mock.results.at(-1)?.value).toBe('loaded-family');
    await vi.waitFor(() => expect(editableIn(host).style.fontFamily).toBe('loaded-family'));

    const previewSignal = ensurePreview.mock.calls[0]?.[1] as AbortSignal;
    act(() => root.unmount());
    mounted.pop();
    host.remove();
    expect(previewSignal.aborted).toBe(true);
  });
});
