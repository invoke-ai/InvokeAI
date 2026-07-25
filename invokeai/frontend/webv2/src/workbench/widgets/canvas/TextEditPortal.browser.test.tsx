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

/**
 * A minimal engine double: the portal only needs the interaction store to read
 * the session, the layers commands it drives, and a viewport for positioning.
 */
const createEngine = (initialSession: TextEditSession | null) => {
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
  };
  const setSession = (next: TextEditSession | null) => {
    current = next;
    listeners.forEach((listener) => listener());
  };
  return { engine: engine as unknown as Parameters<typeof TextEditPortal>[0]['engine'], layers, setSession };
};

const mount = (engine: Parameters<typeof TextEditPortal>[0]['engine']) => {
  const host = window.document.createElement('div');
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
});
