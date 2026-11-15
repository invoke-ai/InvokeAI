import { execFile } from 'node:child_process';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const runFile = promisify(execFile);
const MARKER = '<meta name="invokeai-desktop-theme" content="" />';
const compiler = fileURLToPath(new URL('../../../app/api/desktop_theme.py', import.meta.url));

/** Local preview only: use the backend's value-only compiler without importing the Python application. */
export const desktopThemePlugin = ({ enabled = false, env = process.env } = {}) => ({
  name: 'invokeai-desktop-theme',
  apply: 'serve',
  async transformIndexHtml(html) {
    if (!enabled || !html.includes(MARKER)) {
      return html;
    }
    try {
      const { stdout } = await runFile('python3', [compiler], { env, timeout: 2000, maxBuffer: 2048 });
      return stdout ? html.replace(MARKER, stdout) : html;
    } catch {
      // Python or a local palette is optional; browser system colors remain available.
      return html;
    }
  },
});
