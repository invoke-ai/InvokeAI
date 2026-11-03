import type { TextEditSession } from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop -- the editable's style object is derived from the live session/viewport and intentionally recomputed each render. */
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from 'react';

import { useNotify } from '@workbench/useNotify';
import { useTextEditSession } from '@workbench/widgets/canvas/engineStoreHooks';
import {
  TextFontReadiness,
  textFontKey,
  textFontVariationSettings,
  useResolvedTextFontFamily,
} from '@workbench/widgets/canvas/textFontStyle';
import { reportStructuralCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useRef, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

type TextEditEngine = Pick<CanvasEngineHandle, 'interaction' | 'layers' | 'viewport'> &
  Partial<Pick<CanvasEngineHandle, 'fonts'>>;

/**
 * Render session text as document-unit contenteditable transformed by pan/zoom; the compositor skips that layer.
 * Preserve manual line breaks without wrapping. Keystrokes stay in DOM and stop hotkey propagation; blur/mod+enter
 * commits and Escape cancels.
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

/** Returns keyboard focus to the focusable CanvasSurface container after closing the editor. */
const restoreCanvasFocus = (
  surface: HTMLElement | null,
  editable: HTMLElement,
  ignoreNextBlur: { current: boolean }
): void => {
  if (!(surface instanceof HTMLElement) || surface.tabIndex !== -1) {
    return;
  }

  if (document.activeElement === editable) {
    ignoreNextBlur.current = true;
  }
  surface.focus({ preventScroll: true });
  if (document.activeElement !== surface && ignoreNextBlur.current) {
    ignoreNextBlur.current = false;
  }
};

interface TextEditableProps {
  engine: TextEditEngine;
  session: TextEditSession;
}

/** Key by session.id so content/focus seed once while position and style remain reactive. */
const TextEditable = ({ engine, session }: TextEditableProps) => {
  // Re-render on pan/zoom so the transform below tracks the viewport.
  useViewportTick(engine);
  const viewport = engine.viewport.getViewport();
  const { source, transform } = session;
  const resolvedFontFamily = useResolvedTextFontFamily(engine.fonts, source);
  const ignoreNextBlur = useRef(false);

  // Use a stable ref to seed/focus once and register a live DOM reader for click-away commit without per-key store
  // writes; clear it on unmount.
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

  const notify = useNotify();
  const { t } = useTranslation();
  const commit = useCallback(
    (element: HTMLElement) => {
      const result = engine.layers.commitTextEdit(readEditableText(element));
      if (result) {
        reportStructuralCommit(result, notify.error, t);
      }
    },
    [engine, notify, t]
  );
  const onBlur = useCallback(
    (event: { currentTarget: HTMLElement }) => {
      if (ignoreNextBlur.current) {
        ignoreNextBlur.current = false;
        return;
      }
      commit(event.currentTarget);
    },
    [commit]
  );

  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      // Keep every keystroke inside the field — no canvas hotkey/window key fires.
      event.stopPropagation();
      if (event.key === 'Escape') {
        event.preventDefault();
        const surface = event.currentTarget.parentElement;
        engine.layers.cancelTextEdit();
        if (engine.interaction.get('textEditSession') === null) {
          restoreCanvasFocus(surface, event.currentTarget, ignoreNextBlur);
        }
        return;
      }
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        const surface = event.currentTarget.parentElement;
        commit(event.currentTarget);
        if (engine.interaction.get('textEditSession') === null) {
          restoreCanvasFocus(surface, event.currentTarget, ignoreNextBlur);
        }
      }
    },
    [commit, engine]
  );

  const origin = viewport.documentToScreen({ x: transform.x, y: transform.y });
  const scale = viewport.getZoom() * transform.scaleX;

  const style: CSSProperties = {
    background: 'transparent',
    border: 'none',
    color: source.color,
    cursor: 'text',
    fontFamily: resolvedFontFamily,
    fontSize: `${source.fontSize}px`,
    fontStyle: source.fontStyle ?? 'normal',
    fontVariationSettings: textFontVariationSettings(source) || 'normal',
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
      aria-label={t('widgets.canvas.toolOptions.textEdit')}
      aria-multiline
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
  return (
    <>
      <TextFontReadiness key={textFontKey(session.source)} fonts={engine.fonts} source={session.source} />
      <TextEditable key={session.id} engine={engine} session={session} />
    </>
  );
};
