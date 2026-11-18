/** A 2×2 PNG, for files a test drags in with `commands.nativeDrag`. */
export const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAEklEQVR4nGP4z8DwHwyBNAMDAC+tBv1+4dL7AAAAAElFTkSuQmCC';

declare module 'vitest/browser' {
  interface BrowserCommands {
    /**
     * Drags files, a link or plain text in from outside the page through Chromium's own drag input and drops it at the
     * last point (the test frame's viewport coordinates); defined in `vitest.browser.config.mts`.
     */
    nativeDrag: (options: {
      files?: readonly { base64: string; name: string }[];
      frameWidth: number;
      points: readonly { x: number; y: number }[];
      text?: string;
      uriList?: string;
    }) => Promise<void>;
  }
}
