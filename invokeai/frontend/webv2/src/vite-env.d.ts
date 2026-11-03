/// <reference types="vite/client" />

/** Rewrites canvas pixel goldens instead of comparing them; set by `test:browser:update-goldens`. */
declare const __CANVAS_GOLDEN_UPDATE__: boolean;

interface ImportMetaEnv {
  /** The release version from `invokeai/version/invokeai_version.py`, injected by `vite.config.mts`. */
  readonly APP_VERSION: string;
  /** A content hash of `public/locales`, injected by `scripts/locale-assets-plugin.mjs`. */
  readonly LOCALE_VERSION: string;
}
