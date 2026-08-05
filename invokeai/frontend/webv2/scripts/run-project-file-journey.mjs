import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

import { MOCK_BACKEND_PROFILE_COUNTS, MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME } from './mock-backend-fixtures.mjs';
import { startMockBackend } from './mock-backend.mjs';

const root = resolve(import.meta.dirname, '..');
const port = Number(process.env.INVOKEAI_PROJECT_FILE_PORT ?? 4180);
const origin = `http://127.0.0.1:${String(port)}`;
const backendPort = Number(process.env.INVOKEAI_PROJECT_FILE_BACKEND_PORT ?? 4181);
const backendOrigin = `http://127.0.0.1:${String(backendPort)}`;
const sourceProjectId = 'fixture-project-002';
const sourceProjectName = 'Fixture Project 002';
const sourceProjectPath = `/#/app?project=${sourceProjectId}`;
const journeyTimeoutMs = 60_000;
const cleanupTimeoutMs = 2_000;

const delay = (durationMs) =>
  new Promise((resolveDelay) => {
    setTimeout(resolveDelay, durationMs);
  });

const fetchJson = async (path, init) => {
  const response = await fetch(`${backendOrigin}${path}`, init);
  const body = await response.json();

  assert.equal(
    response.ok,
    true,
    `${init?.method ?? 'GET'} ${path} returned ${String(response.status)}: ${JSON.stringify(body)}`
  );

  return body;
};

const waitForPreview = async (getPreviewExit) => {
  const deadline = Date.now() + 20_000;

  while (Date.now() < deadline) {
    const previewExit = getPreviewExit();

    if (previewExit !== null) {
      throw new Error(`Vite preview exited before becoming ready (${previewExit}).`);
    }

    try {
      const response = await fetch(origin);

      if (response.ok) {
        return;
      }
    } catch {
      // Preview is still starting.
    }

    await delay(100);
  }

  throw new Error(`Vite preview did not become ready at ${origin}.`);
};

const observeBrowserErrors = (page, phase, errors) => {
  page.on('pageerror', (error) => {
    errors.push(new Error(`${phase} page error: ${error.stack ?? error.message}`));
  });
  page.on('console', (message) => {
    if (message.type() === 'error') {
      const location = message.location();

      // Every release harness deliberately uses the HTTP-only mock backend, so
      // Socket.IO's transport probe is expected to stay disconnected. Ignore
      // only that browser-generated 404; application console errors still fail.
      if (
        location.url.includes('/ws/socket.io/') &&
        message.text().includes('the server responded with a status of 404')
      ) {
        return;
      }

      // Video existence has no bulk endpoint. Import deliberately probes the
      // archived name and treats this exact 404 as the signal to restore it.
      if (
        phase === 'import' &&
        location.url &&
        new URL(location.url).pathname === `/api/v1/videos/i/${MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME}` &&
        message.text().includes('the server responded with a status of 404')
      ) {
        return;
      }

      const where = location.url ? ` (${location.url}:${String(location.lineNumber)})` : '';

      errors.push(new Error(`${phase} console error${where}: ${message.text()}`));
    }
  });
};

const assertNoBrowserErrors = (errors) => {
  if (errors.length > 0) {
    throw new AggregateError(errors, 'The project-file journey raised browser errors.');
  }
};

const assertAssetRoute = async (basePath, name, variants) => {
  const encodedName = encodeURIComponent(name);
  const dto = await fetch(`${backendOrigin}${basePath}/i/${encodedName}`);

  assert.equal(dto.status, 200, `${basePath}/i/${encodedName} returned ${String(dto.status)}.`);

  for (const variant of variants) {
    const asset = await fetch(`${backendOrigin}${basePath}/i/${encodedName}/${variant}`);

    assert.equal(asset.status, 200, `${basePath}/i/${encodedName}/${variant} returned ${String(asset.status)}.`);
    assert.ok((await asset.arrayBuffer()).byteLength > 0, `${basePath}/${variant} returned an empty body.`);
  }
};

const waitForProjectCover = async (projectId, restoredImageNames) => {
  const deadline = Date.now() + 10_000;

  while (Date.now() < deadline) {
    const raw = await fetchJson('/api/v1/client_state/default/get_by_key?key=webv2%3Aproject-covers');

    if (typeof raw === 'string') {
      const coverIndex = JSON.parse(raw);
      const coverImageName = coverIndex[projectId];

      if (restoredImageNames.has(coverImageName)) {
        return coverImageName;
      }
    }

    await delay(50);
  }

  assert.fail(`The project cover index never mapped ${projectId} to a restored image.`);
};

const runRoundTrip = async ({ backend, browser, contexts, errors, tempDirectory }) => {
  const exportContext = await browser.newContext({ acceptDownloads: true });

  contexts.add(exportContext);
  const exportPage = await exportContext.newPage();
  observeBrowserErrors(exportPage, 'export', errors);
  await exportPage.goto(`${origin}${sourceProjectPath}`, { waitUntil: 'domcontentloaded' });
  await exportPage.getByRole('main', { exact: true, name: sourceProjectName }).waitFor();

  const downloadPromise = exportPage.waitForEvent('download');
  await exportPage
    .getByRole('button', { exact: true, name: `Switch project. Current project: ${sourceProjectName}` })
    .click();
  await exportPage.getByRole('menuitem', { exact: true, name: 'Export' }).click();
  const download = await downloadPromise;
  const archivePath = join(tempDirectory, 'fixture-project-002.invk');

  assert.equal(download.suggestedFilename(), `${sourceProjectName}.invk`);
  await download.saveAs(archivePath);
  await exportContext.close();
  contexts.delete(exportContext);
  assertNoBrowserErrors(errors);

  const reset = await fetchJson('/__reset?profile=empty', { method: 'POST' });

  assert.equal(reset.profile, 'empty');
  assert.deepEqual(reset.counts, MOCK_BACKEND_PROFILE_COUNTS.empty);
  assert.equal(backend.profile(), 'empty');

  const importContext = await browser.newContext();

  contexts.add(importContext);
  const importPage = await importContext.newPage();
  const uploadResponses = [];

  importPage.on('response', (response) => {
    const path = new URL(response.url()).pathname;
    const kind = path === '/api/v1/images/upload' ? 'image' : path === '/api/v1/videos/upload' ? 'video' : null;

    if (kind !== null && response.request().method() === 'POST') {
      uploadResponses.push(
        response.json().then((dto) => ({
          kind,
          name: kind === 'image' ? dto.image_name : dto.video_name,
          status: response.status(),
        }))
      );
    }
  });
  observeBrowserErrors(importPage, 'import', errors);
  await importPage.goto(`${origin}/#/`, { waitUntil: 'domcontentloaded' });
  await importPage.getByRole('heading', { exact: true, name: 'Welcome to Invoke' }).waitFor();

  const chooserPromise = importPage.waitForEvent('filechooser');
  await importPage.getByRole('button', { exact: true, name: 'Import…' }).click();
  const chooser = await chooserPromise;

  await chooser.setFiles(archivePath);
  await importPage.waitForURL(/#\/app\?project=/);
  await importPage.getByRole('main', { exact: true, name: sourceProjectName }).waitFor();

  const projects = await fetchJson('/api/v1/projects/');

  assert.equal(projects.length, 1);
  const [summary] = projects;

  assert.ok(summary);
  assert.notEqual(summary.project_id, sourceProjectId);
  assert.equal(importPage.url().includes(`project=${encodeURIComponent(summary.project_id)}`), true);

  const imported = await fetchJson(`/api/v1/projects/${encodeURIComponent(summary.project_id)}`);
  const images = await fetchJson('/api/v1/images/?categories=other&limit=100&offset=0');
  const videos = await fetchJson('/api/v1/videos/?categories=other&limit=100&offset=0');
  const uploads = await Promise.all(uploadResponses);

  assert.equal(imported.project_id, summary.project_id);
  assert.equal(imported.data.id, summary.project_id);
  assert.equal(images.total, 4);
  assert.equal(images.items.length, 4);
  assert.equal(videos.total, 1);
  assert.equal(videos.items.length, 1);

  const restoredImageNames = new Set(images.items.map((image) => image.image_name));
  const restoredVideoNames = new Set(videos.items.map((video) => video.video_name));
  const uploadedImageNames = uploads.filter(({ kind }) => kind === 'image').map(({ name }) => name);
  const uploadedVideoNames = uploads.filter(({ kind }) => kind === 'video').map(({ name }) => name);
  const layerImageNames = imported.data.canvas.document.layers.map((layer) => layer.source.image.imageName);
  const videoName = imported.data.projectGraph.nodes[0]?.data.inputs.video?.value?.video_name;

  assert.equal(
    uploads.every(({ status }) => status === 201),
    true
  );
  assert.equal(uploadedImageNames.length, 4);
  assert.equal(uploadedVideoNames.length, 1);
  assert.deepEqual([...uploadedImageNames].sort(), [...restoredImageNames].sort());
  assert.deepEqual(uploadedVideoNames, [...restoredVideoNames]);
  assert.equal(layerImageNames.length, 4);
  assert.equal(new Set(layerImageNames).size, 4);
  assert.deepEqual([...layerImageNames].sort(), [...restoredImageNames].sort());
  assert.deepEqual([...restoredVideoNames], [videoName]);
  assert.notEqual(videoName, MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME);

  for (const image of images.items) {
    assert.equal(image.image_category, 'other');
    assert.equal(image.is_intermediate, false);
    await assertAssetRoute('/api/v1/images', image.image_name, ['full', 'thumbnail']);
  }

  for (const video of videos.items) {
    assert.equal(video.video_category, 'other');
    assert.equal(video.is_intermediate, false);
    await assertAssetRoute('/api/v1/videos', video.video_name, ['full', 'thumbnail']);
  }

  const coverImageName = await waitForProjectCover(summary.project_id, restoredImageNames);

  assert.ok(layerImageNames.includes(coverImageName));
  await importContext.close();
  contexts.delete(importContext);
  assertNoBrowserErrors(errors);

  return {
    coverImageName,
    imageNames: [...restoredImageNames].sort(),
    projectId: summary.project_id,
    videoName,
  };
};

const toError = (error) => (error instanceof Error ? error : new Error(String(error)));

const withTimeout = async (run, durationMs, label) => {
  let timer;

  try {
    return await Promise.race([
      Promise.resolve().then(run),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} exceeded ${String(durationMs)} ms.`)), durationMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};

const getDefaultDependencies = () => ({
  createTempDirectory: () => mkdtemp(join(tmpdir(), 'invokeai-project-file-journey-')),
  killProcessGroup: (pid, signal) => process.kill(-pid, signal),
  launchBrowser: ({ timeoutMs }) => chromium.launch({ headless: true, timeout: timeoutMs }),
  now: () => performance.now(),
  removeTempDirectory: (directory) => rm(directory, { force: true, recursive: true }),
  runRoundTrip,
  spawnPreview: () =>
    spawn('pnpm', ['exec', 'vite', 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
      cwd: root,
      detached: true,
      env: { ...process.env, INVOKEAI_DEV_BACKEND: backendOrigin },
      stdio: ['ignore', 'ignore', 'pipe'],
    }),
  startBackend: () => startMockBackend(backendPort, { profile: 'representative' }),
  waitForPreview: ({ getPreviewExit }) => waitForPreview(getPreviewExit),
});

/**
 * Owns the complete journey deadline and every disposable resource. Dependencies
 * are injectable so timeout and teardown behavior can be tested without
 * launching a browser or binding a port.
 */
export const executeProjectFileJourney = async ({
  cleanupTimeoutMs: teardownLimitMs = cleanupTimeoutMs,
  dependencies: dependencyOverrides = {},
  timeoutMs = journeyTimeoutMs,
} = {}) => {
  const dependencies = { ...getDefaultDependencies(), ...dependencyOverrides };
  const startedAt = dependencies.now();
  const deadlineAt = startedAt + timeoutMs;
  const controller = new AbortController();
  const contexts = new Set();
  const browserErrors = [];
  const cleanupErrors = [];
  const pendingOperations = new Set();
  let backend = null;
  let browser = null;
  let preview = null;
  let previewError = '';
  let previewExit = null;
  let tempDirectory = null;
  let primaryFailure = null;
  let teardownPromise = null;
  let isJourneyComplete = false;

  const remainingMs = () => Math.max(1, Math.ceil(deadlineAt - dependencies.now()));
  const recordCleanupError = (label, error) => {
    cleanupErrors.push(new Error(`${label}: ${toError(error).message}`, { cause: error }));
  };
  const attemptCleanup = async (label, run) => {
    try {
      await withTimeout(run, teardownLimitMs, `${label} cleanup`);
    } catch (error) {
      recordCleanupError(label, error);
    }
  };
  const trackOperation = (operation) => {
    pendingOperations.add(operation);
    void operation.then(
      () => pendingOperations.delete(operation),
      () => pendingOperations.delete(operation)
    );

    return operation;
  };
  const waitForAbort = (operation) =>
    new Promise((resolve, reject) => {
      const onAbort = () => reject(controller.signal.reason);

      if (controller.signal.aborted) {
        reject(controller.signal.reason);
        return;
      }

      controller.signal.addEventListener('abort', onAbort, { once: true });
      operation.then(
        (value) => {
          controller.signal.removeEventListener('abort', onAbort);
          resolve(value);
        },
        (error) => {
          controller.signal.removeEventListener('abort', onAbort);
          reject(error);
        }
      );
    });
  const acquire = async (label, create, dispose, assign) => {
    const acquisition = trackOperation(
      Promise.resolve()
        .then(() => create({ signal: controller.signal, timeoutMs: remainingMs() }))
        .then(async (resource) => {
          if (controller.signal.aborted || teardownPromise !== null) {
            await attemptCleanup(`${label} created after teardown`, () => dispose(resource));
            throw controller.signal.reason ?? new Error(`${label} completed after teardown.`);
          }

          assign(resource);
          return resource;
        })
    );

    return waitForAbort(acquisition);
  };
  const stopPreview = async () => {
    if (!preview?.pid || previewExit !== null) {
      return;
    }

    const exitGraceMs = Math.min(500, Math.max(1, Math.floor(teardownLimitMs / 3)));
    const waitForExit = () =>
      new Promise((resolveExit) => {
        if (previewExit !== null) {
          resolveExit(true);
          return;
        }

        const onExit = () => {
          clearTimeout(timer);
          resolveExit(true);
        };
        const timer = setTimeout(() => {
          preview.removeListener('exit', onExit);
          resolveExit(false);
        }, exitGraceMs);

        preview.once('exit', onExit);
      });

    try {
      dependencies.killProcessGroup(preview.pid, 'SIGTERM');
    } catch (error) {
      if (error?.code !== 'ESRCH') {
        throw error;
      }
      return;
    }

    if (await waitForExit()) {
      return;
    }

    try {
      dependencies.killProcessGroup(preview.pid, 'SIGKILL');
    } catch (error) {
      if (error?.code !== 'ESRCH') {
        throw error;
      }
      return;
    }

    if (!(await waitForExit())) {
      throw new Error('Vite preview did not exit after SIGKILL.');
    }
  };
  const teardown = () => {
    if (teardownPromise !== null) {
      return teardownPromise;
    }

    teardownPromise = (async () => {
      if (!controller.signal.aborted) {
        controller.abort(new Error('Project-file journey teardown started.'));
      }

      const resourceCleanups = [
        ...[...contexts].map((context, index) =>
          attemptCleanup(`browser context ${String(index + 1)}`, () => context.close())
        ),
        ...(browser === null ? [] : [attemptCleanup('browser', () => browser.close())]),
        ...(preview === null ? [] : [attemptCleanup('Vite preview', stopPreview)]),
        ...(backend === null ? [] : [attemptCleanup('mock backend', () => backend.close())]),
      ];

      await Promise.all(resourceCleanups);
      contexts.clear();

      const pending = [...pendingOperations];
      if (pending.length > 0) {
        await attemptCleanup('pending journey operations', () => Promise.allSettled(pending));
      }

      if (tempDirectory !== null) {
        const ownedTempDirectory = tempDirectory;
        tempDirectory = null;
        await attemptCleanup('journey temp directory', () => dependencies.removeTempDirectory(ownedTempDirectory));
      }
    })();

    return teardownPromise;
  };
  const fail = (error) => {
    if (primaryFailure === null) {
      primaryFailure = toError(error);
    }
    if (!controller.signal.aborted) {
      controller.abort(primaryFailure);
    }
    void teardown();
  };
  const deadlineTimer = setTimeout(() => {
    fail(new Error(`Project-file journey exceeded its ${String(timeoutMs / 1_000)}-second timeout.`));
  }, timeoutMs);
  let result;

  try {
    await acquire(
      'journey temp directory',
      () => dependencies.createTempDirectory(),
      (directory) => dependencies.removeTempDirectory(directory),
      (directory) => {
        tempDirectory = directory;
      }
    );
    await acquire(
      'mock backend',
      ({ signal }) => dependencies.startBackend({ signal }),
      (resource) => resource.close(),
      (resource) => {
        backend = resource;
      }
    );

    preview = dependencies.spawnPreview({ signal: controller.signal });
    preview.stderr?.on('data', (chunk) => {
      previewError += String(chunk);
    });
    preview.on('error', (error) => fail(error));
    preview.on('exit', (code, signal) => {
      previewExit = signal ? `signal ${signal}` : `code ${String(code)}`;
      if (teardownPromise === null && !isJourneyComplete) {
        fail(new Error(`Vite preview exited during the journey (${previewExit}).`));
      }
    });

    await waitForAbort(
      trackOperation(
        Promise.resolve(
          dependencies.waitForPreview({
            getPreviewExit: () => previewExit,
            signal: controller.signal,
          })
        )
      )
    );
    await acquire(
      'browser',
      ({ signal, timeoutMs: setupTimeoutMs }) =>
        dependencies.launchBrowser({ errors: browserErrors, signal, timeoutMs: setupTimeoutMs }),
      (resource) => resource.close(),
      (resource) => {
        browser = resource;
      }
    );

    result = await waitForAbort(
      trackOperation(
        Promise.resolve(
          dependencies.runRoundTrip({
            backend,
            browser,
            contexts,
            errors: browserErrors,
            signal: controller.signal,
            tempDirectory,
          })
        )
      )
    );
    isJourneyComplete = true;
  } catch (error) {
    fail(error);
  } finally {
    clearTimeout(deadlineTimer);
    await teardown();
  }

  const durationMs = Math.round(dependencies.now() - startedAt);
  const failures = [...(primaryFailure === null ? [] : [primaryFailure]), ...cleanupErrors, ...browserErrors];

  if (failures.length > 0) {
    const detail = failures.map((error) => error.stack ?? error.message).join('\n');
    const previewDetail = previewError ? `\nVite preview stderr:\n${previewError}` : '';

    throw new Error(`${detail}${previewDetail}\nJourney duration: ${String(durationMs)} ms.`, {
      cause: primaryFailure ?? failures[0],
    });
  }

  return { durationMs, result };
};

const isMain = process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) {
  const { durationMs, result } = await executeProjectFileJourney();

  process.stdout.write(
    `${JSON.stringify(
      {
        durationMs,
        ports: { backend: backendPort, preview: port },
        result,
        status: 'passed',
        timeoutMs: journeyTimeoutMs,
      },
      null,
      2
    )}\n`
  );
}
