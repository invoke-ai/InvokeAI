/**
 * Web-font readiness for text layers.
 *
 * A text layer's rasterized metrics depend on the font actually being available:
 * if a web font isn't loaded yet, the platform substitutes a fallback and
 * `measureText` reports the WRONG advances, so the cached text pixels (and the
 * layer's extent) would be wrong until something else happened to re-rasterize.
 *
 * This loader is the seam the engine uses to fix that without importing the DOM
 * into the rasterizer: given a `FontLoadApi` (the browser's `document.fonts`, or
 * a fake in tests), it kicks a load for a not-yet-available font exactly once
 * (concurrent requests for the same font string share one in-flight promise) and
 * invokes `onReady` when it resolves — the engine's callback re-rasterizes the
 * affected layer. A `null` api (node, or a browser without `document.fonts`) is
 * a silent no-op, so importing this module and the engine stays node-safe.
 *
 * The loader holds NO permanent "loaded" set — it trusts `api.check()` (the
 * browser flips it to `true` post-load) as the source of truth and only tracks
 * in-flight promises (cleared on settle). Combined with the per-engine instance,
 * that keeps it free of cross-test global state.
 *
 * Zero React, zero import-time side effects.
 */

/** The minimal slice of `FontFaceSet` the loader needs (so tests can inject a fake). */
export interface FontLoadApi {
  /** Whether every face for the CSS `font` shorthand is loaded and usable. */
  check(font: string): boolean;
  /** Loads the faces for the CSS `font` shorthand; resolves when they're ready. */
  load(font: string): Promise<unknown>;
}

/** A per-engine font loader bound to one `FontLoadApi` (or `null` in node). */
export interface FontLoader {
  /**
   * Ensures `font` is available, invoking `onReady` once when a pending load
   * resolves. A no-op (never calls `onReady`) when there is no api or the font
   * is already available — the caller's synchronous rasterize already used the
   * correct metrics in that case.
   */
  ensure(font: string, onReady: () => void): void;
}

/** `check` can throw on an unparseable font string; treat a throw as "not available". */
const safeCheck = (api: FontLoadApi, font: string): boolean => {
  try {
    return api.check(font);
  } catch {
    return false;
  }
};

/** Creates a font loader bound to `api` (pass `null` for a node-safe no-op loader). */
export const createFontLoader = (api: FontLoadApi | null): FontLoader => {
  const pending = new Map<string, Promise<void>>();

  return {
    ensure: (font, onReady) => {
      if (!api || safeCheck(api, font)) {
        return;
      }
      const existing = pending.get(font);
      if (existing) {
        existing.then(onReady, () => {});
        return;
      }
      const promise = Promise.resolve(api.load(font)).then(
        () => {},
        () => {}
      );
      const tracked = promise.finally(() => pending.delete(font));
      pending.set(font, tracked);
      tracked.then(onReady, () => {});
    },
  };
};

/** Resolves the browser's `document.fonts` api, or `null` where it is unavailable (node, older browsers). */
export const domFontLoadApi = (): FontLoadApi | null => {
  if (typeof document === 'undefined') {
    return null;
  }
  const fonts = (document as Document & { fonts?: FontLoadApi }).fonts;
  return fonts && typeof fonts.check === 'function' && typeof fonts.load === 'function' ? fonts : null;
};
