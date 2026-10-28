/**
 * `pnpm graphs:export <request.json> <response.json>` compiles offline export cases into the enqueue batches the UI
 * would send (src/app/graphExport.ts). Vite's module runner loads the TypeScript sources with the app's aliases and
 * opens no port. Exit 0 once the response is written, with per-case errors inside it; exit 2 for an unreadable or
 * malformed request, with no response written.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { loadConfigFromFile, runnerImport } from 'vite';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const EXIT_MALFORMED_REQUEST = 2;

const rejectRequest = (message) => {
  process.stderr.write(`graphs:export: ${message}\n`);
  process.exitCode = EXIT_MALFORMED_REQUEST;
};

const readRequest = async (path) => {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    rejectRequest(`cannot read ${path}: ${error instanceof Error ? error.message : String(error)}`);
    return undefined;
  }
};

/** Only the aliases and defines: plugins transform the browser bundle, not these pure modules. */
const loadExporter = async () => {
  const { config } = await loadConfigFromFile(
    { command: 'serve', mode: 'development' },
    resolve(PACKAGE_ROOT, 'vite.config.mts'),
    PACKAGE_ROOT,
    'error'
  );
  const { module } = await runnerImport(resolve(PACKAGE_ROOT, 'src/app/graphExport.ts'), {
    define: config.define,
    logLevel: 'error',
    resolve: { alias: config.resolve.alias },
    root: PACKAGE_ROOT,
  });

  return module;
};

const main = async () => {
  const [requestPath, responsePath, ...extra] = process.argv.slice(2);

  if (!requestPath || !responsePath || extra.length > 0) {
    rejectRequest('usage: pnpm graphs:export <request.json> <response.json>');
    return;
  }

  const request = await readRequest(requestPath);

  if (request === undefined) {
    return;
  }

  const { exportGraphs, GraphExportRequestError } = await loadExporter();
  let response;

  try {
    response = exportGraphs(request);
  } catch (error) {
    if (error instanceof GraphExportRequestError) {
      rejectRequest(`malformed request: ${error.message}`);
      return;
    }

    throw error;
  }

  await writeFile(responsePath, `${JSON.stringify(response, null, 2)}\n`);
};

await main();
