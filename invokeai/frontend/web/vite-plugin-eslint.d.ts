declare module 'vite-plugin-eslint' {
  import type { ESLint } from 'eslint';
  import type { Plugin } from 'vite';

  interface Options extends ESLint.Options {
    eslintPath?: string;
    lintOnStart?: boolean;
    include?: string | string[];
    exclude?: string | string[];
    formatter?: string | ESLint.Formatter['format'];
    emitWarning?: boolean;
    emitError?: boolean;
    failOnWarning?: boolean;
    failOnError?: boolean;
  }

  export default function eslintPlugin(rawOptions?: Options): Plugin;
}
