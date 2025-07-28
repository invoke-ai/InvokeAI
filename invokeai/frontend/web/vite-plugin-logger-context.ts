import MagicString from 'magic-string';
import type { Plugin } from 'vite';

/**
 * A Vite plugin that automatically adds file path context to logger calls.
 */
export function loggerContextPlugin(): Plugin {
  return {
    name: 'logger-context',
    transform(code: string, id: string) {
      // Only process TypeScript/JavaScript files in src directory
      if (!id.includes('/src/') || !id.match(/\.(ts|tsx|js|jsx)$/)) {
        return null;
      }

      // Check if the file imports logger
      if (!code.includes("from 'app/logging/logger'") && !code.includes('from "app/logging/logger"')) {
        return null;
      }

      const s = new MagicString(code);

      // Extract relative path from src/
      const srcIndex = id.indexOf('/src/');
      const relativePath = srcIndex !== -1 ? id.substring(srcIndex + 5) : id.split('/').pop() || 'unknown';

      // Match logger calls: logger('namespace')
      const loggerRegex = /\blogger\s*\(\s*['"`](\w+)['"`]\s*\)/g;
      let match;

      while ((match = loggerRegex.exec(code)) !== null) {
        const fullMatch = match[0];
        const namespace = match[1];
        const startIndex = match.index;
        const endIndex = startIndex + fullMatch.length;

        // Replace with logger('namespace').child({ filePath: 'path/to/file.ts' })
        s.overwrite(startIndex, endIndex, `logger('${namespace}').child({ filePath: '${relativePath}' })`);
      }

      if (s.hasChanged()) {
        return {
          code: s.toString(),
          map: s.generateMap({ hires: true }),
        };
      }

      return null;
    },
  };
}
