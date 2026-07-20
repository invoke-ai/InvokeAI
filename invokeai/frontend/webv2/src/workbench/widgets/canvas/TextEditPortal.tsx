import type { TextEditSession } from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop -- the editable's style object is derived from the live session/viewport and intentionally recomputed each render. */
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from 'react';

import { useTextEditSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback, useSyncExternalStore } from 'react';

type TextEditEngine = Pick<CanvasEngineHandle, 'interaction' | 'layers' | 'viewport'>;

/**
 * The text-editing portal: a positioned `contenteditable` div, rendered over the
 * canvas whenever a text-edit session is active, that IS the text while editing
 * (the compositor skips the session's layer so the two never double-draw).
 *
 * ## WYSIWYG
 *
 * The editable's intrinsic styles are set in DOCUMENT units (font size in px,
 * unitless line-height, family/weight/align/color from the live session source)
 * — exactly what the rasterizer will bake. A single CSS transform then maps the
 * layer's document anchor to the screen and magnifies by the view zoom, so the
 * on-screen editable matches the rasterized output at any pan/zoom. `white-space:
 * pre` mirrors the rasterizer's manual-line-break, no-auto-wrap layout.
 *
 * ## Keystrokes stay local
 *
 * Typing only mutates the editable's DOM — never the engine/store (no per-key
 * traffic). Every keydown is `stopPropagation`'d so canvas hotkeys can't fire
 * from the field (belt-and-braces with the pipeline's editable guard and the
 * widget hotkeys' `allowInEditable: false`). Commit is on blur / `mod+enter`;
 * `esc` cancels — the editable owns Escape (stopPropagation) so the engine's
 * Escape priority never also runs.
 */

/** Re-renders on any viewport (pan/zoom) change via a value-stable snapshot string. */
const useViewportTick = (engine: TextEditEngine): string => {
  const viewport = engine.viewport.getViewport();
  const subscribe = useCallback((onChange: () => void) => viewport.subscribe(onChange), [viewport]);
  const getSnapshot = useCallback(() => {
    const { pan, zoom } = viewport.getState();
    return `${zoom}|${pan.x}|${pan.y}`;
  }, [viewport]);
  useSyncExternalStore(subscribe, getSnapshot);
  return '';
};

/** Reads the editable's text with manual line breaks preserved (`\n` per visual line). */
const readEditableText = (el: HTMLElement): string => el.innerText;

/** Moves the caret to the end of `el`'s content. */
const placeCaretAtEnd = (el: HTMLElement): void => {
  const selection = window.getSelection();
  if (!selection) {
    return;
  }
  const range = document.createRange();
  range.selectNodeContents(el);
  range.collapse(false);
  selection.removeAllRanges();
  selection.addRange(range);
};

interface TextEditableProps {
  engine: TextEditEngine;
  session: TextEditSession;
}

/**
 * The single editable element for one session. Keyed by `session.id` in the
 * parent so a fresh session remounts it — the ref callback then seeds content +
 * focus exactly once, while position/style recompute on each render.
 */
const TextEditable = ({ engine, session }: TextEditableProps) => {
  // Re-render on pan/zoom so the transform below tracks the viewport.
  useViewportTick(engine);
  const viewport = engine.viewport.getViewport();
  const { source, transform } = session;

  // Seeds content + focus once when the element mounts, and registers a live-
  // content reader with the engine so it can commit on a canvas pointerdown
  // (click-elsewhere-to-commit) without per-keystroke traffic. Cleared on unmount.
  // A stable callback, so React never re-runs it on a position/style re-render.
  const setRef = useCallback(
    (el: HTMLDivElement | null) => {
      if (!el) {
        // Unmount: stop the engine from reading a detached element.
        engine.layers.setTextEditContentReader(null);
        return;
      }
      engine.layers.setTextEditContentReader(() => readEditableText(el));
      if (el.dataset.seeded === 'true') {
        return;
      }
      el.dataset.seeded = 'true';
      el.textContent = source.content;
      el.focus();
      placeCaretAtEnd(el);
    },
    // `session` is stable for this element's life (parent keys by session.id);
    // seed from the source captured at mount.
    [engine, source.content]
  );

  const onBlur = useCallback(
    (event: { currentTarget: HTMLElement }) => {
      engine.layers.commitTextEdit(readEditableText(event.currentTarget));
    },
    [engine]
  );

  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      // Keep every keystroke inside the field — no canvas hotkey/window key fires.
      event.stopPropagation();
      if (event.key === 'Escape') {
        event.preventDefault();
        engine.layers.cancelTextEdit();
        return;
      }
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        engine.layers.commitTextEdit(readEditableText(event.currentTarget));
      }
    },
    [engine]
  );

  const origin = viewport.documentToScreen({ x: transform.x, y: transform.y });
  const scale = viewport.getZoom() * transform.scaleX;

  const style: CSSProperties = {
    background: 'transparent',
    border: 'none',
    color: source.color,
    cursor: 'text',
    fontFamily: source.fontFamily,
    fontSize: `${source.fontSize}px`,
    fontWeight: source.fontWeight,
    left: 0,
    lineHeight: source.lineHeight,
    margin: 0,
    minWidth: '1ch',
    outline: 'none',
    padding: 0,
    pointerEvents: 'auto',
    position: 'absolute',
    textAlign: source.align,
    top: 0,
    transform: `translate(${origin.x}px, ${origin.y}px) rotate(${transform.rotation}rad) scale(${scale})`,
    transformOrigin: '0 0',
    whiteSpace: 'pre',
    zIndex: 4,
  };

  return (
    <div
      aria-label="Text editor"
      contentEditable
      dir="auto"
      ref={setRef}
      role="textbox"
      style={style}
      suppressContentEditableWarning
      tabIndex={0}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
    />
  );
};

/** Renders the editable for the active session, or nothing. */
export const TextEditPortal = ({ engine }: { engine: TextEditEngine }) => {
  const session = useTextEditSession(engine);
  if (!session) {
    return null;
  }
  return <TextEditable key={session.id} engine={engine} session={session} />;
};
