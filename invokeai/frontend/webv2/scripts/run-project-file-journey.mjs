import { unzipSync, zipSync } from 'fflate';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

import {
  collectCanvasLeaves,
  MOCK_BACKEND_PROFILE_COUNTS,
  MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME,
  PROJECT_FILE_BOARD,
  PROJECT_FILE_BOARD_ID,
} from './mock-backend-fixtures.mjs';
import { startMockBackend } from './mock-backend.mjs';
import { killPreview, spawnPreview } from './preview-server.mjs';

const root = resolve(import.meta.dirname, '..');
const port = Number(process.env.INVOKEAI_PROJECT_FILE_PORT ?? 4180);
const origin = `http://127.0.0.1:${String(port)}`;
const backendPort = Number(process.env.INVOKEAI_PROJECT_FILE_BACKEND_PORT ?? 4181);
const backendOrigin = `http://127.0.0.1:${String(backendPort)}`;
const sourceProjectId = 'fixture-project-002';
const sourceProjectName = 'Fixture Project 002';
const sourceProjectPath = `/#/app?project=${sourceProjectId}`;
const journeyTimeoutMs = 120_000;
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

      // The HTTP-only mock has no Socket.IO; ignore only its expected transport 404.
      if (
        location.url.includes('/ws/socket.io/') &&
        message.text().includes('the server responded with a status of 404')
      ) {
        return;
      }

      // Import probes archived video names; this 404 triggers restoration.
      if (
        phase.endsWith('import') &&
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

/** Compare membership attributes, excluding names because board media receives new identities on import. */
const boardShape = (items) => items.map((item) => `${item.kind}:${item.category}:${String(item.starred)}`).sort();

const EXPECTED_BOARD_SHAPE = [
  'image:general:false',
  'image:general:false',
  'image:general:true',
  'image:mask:false',
  'image:user:false',
  'video:general:false',
].sort();

/** The archived names of everything on the source board, for proving none of them is reused. */
const ARCHIVED_BOARD_NAMES = [
  PROJECT_FILE_BOARD.referencedImage,
  PROJECT_FILE_BOARD.unreferencedImage,
  PROJECT_FILE_BOARD.starredImage,
  PROJECT_FILE_BOARD.userAsset,
  PROJECT_FILE_BOARD.maskAsset,
  PROJECT_FILE_BOARD.video,
];

const readArchiveEntries = async (archivePath) => unzipSync(new Uint8Array(await readFile(archivePath)));

const readEntryJson = (entries, name) => {
  assert.ok(entries[name], `The archive has no ${name}.`);

  return JSON.parse(new TextDecoder().decode(entries[name]));
};

/** Rewrite an archive without the named entries, to exercise what a damaged one reports. */
const writeArchiveWithout = async (entries, omitted, targetPath) => {
  for (const name of omitted) {
    assert.ok(entries[name], `Cannot omit ${name}: the archive does not carry it.`);
  }

  const kept = Object.fromEntries(Object.entries(entries).filter(([name]) => !omitted.includes(name)));

  await writeFile(targetPath, Buffer.from(zipSync(kept)));

  return targetPath;
};

const getBoardSnapshot = (projectId) => fetchJson(`/api/v1/projects/${encodeURIComponent(projectId)}/board-snapshot`);

const getLayerImageNames = (project) =>
  collectCanvasLeaves(project.data.canvas.document).map((layer) => layer.source.image.imageName);

/** The first workflow document of a project record, whichever document schema the record carries. */
const getWorkflowDocuments = (project) =>
  project.data.workflows?.entries?.map((entry) => entry.document) ?? [project.data.projectGraph].filter(Boolean);

const getDocumentVideoName = (project) =>
  getWorkflowDocuments(project)[0]?.nodes[0]?.data.inputs.video?.value?.video_name;

const workflowRequests = async () => (await fetchJson('/__workflow-requests')).requests;

const presetStrip = (page) => page.getByRole('tablist', { exact: true, name: 'Layout preset' });

const centerViewTrigger = (page, label) => page.getByRole('button', { exact: true, name: `Center view: ${label}` });

/** Preset names need not match their center views; pass the expected view explicitly. */
const selectLayoutPreset = async (page, preset, centerView) => {
  const name = new RegExp(`^${preset}(, unsaved changes)?$`);
  const strip = presetStrip(page);
  const selected = strip.getByRole('tab', { name, selected: true });

  if ((await selected.count()) === 0) {
    await strip.getByRole('tab', { name }).click();
  }

  await selected.waitFor();
  await centerViewTrigger(page, centerView).waitFor();
};

/** Poll the server record until the project's saved workflows satisfy `predicate`. */
const waitForSavedWorkflows = async (projectId, predicate, label) => {
  const deadline = Date.now() + 15_000;

  while (Date.now() < deadline) {
    const record = await fetchJson(`/api/v1/projects/${encodeURIComponent(projectId)}`);

    if (predicate(record)) {
      return record;
    }

    await delay(100);
  }

  throw new Error(`The project record never reached the expected state: ${label}.`);
};

/** Import the archive at `archivePath` through the Launchpad, exactly as a person would. */
const importArchive = async ({ archivePath, browser, contexts, errors, phase }) => {
  const context = await browser.newContext();

  contexts.add(context);
  const page = await context.newPage();

  observeBrowserErrors(page, phase, errors);
  await page.goto(`${origin}/#/`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { exact: true, name: 'Welcome to Invoke' }).waitFor();

  const chooserPromise = page.waitForEvent('filechooser');

  await page.getByRole('button', { exact: true, name: 'Import…' }).click();
  await (await chooserPromise).setFiles(archivePath);
  await page.waitForURL(/#\/app$/);
  await page.getByRole('main', { exact: true, name: sourceProjectName }).waitFor();

  return { context, page };
};

/**
 * Verify imports copy board media despite name collisions, reuse external references, and exclude
 * intermediate/other items.
 */
const runRoundTrip = async ({ backend, browser, contexts, errors, tempDirectory }) => {
  const sourceBoard = await getBoardSnapshot(sourceProjectId);

  assert.deepEqual(boardShape(sourceBoard.items), EXPECTED_BOARD_SHAPE);

  const exportContext = await browser.newContext({ acceptDownloads: true });

  contexts.add(exportContext);
  const exportPage = await exportContext.newPage();

  observeBrowserErrors(exportPage, 'export', errors);
  await exportPage.goto(`${origin}${sourceProjectPath}`, { waitUntil: 'domcontentloaded' });
  await exportPage.getByRole('main', { exact: true, name: sourceProjectName }).waitFor();

  const selectedBoardRow = exportPage.locator('button[aria-current="true"]').filter({ hasText: sourceProjectName });

  await selectedBoardRow.first().waitFor();

  await exportPage.getByRole('button', { exact: true, name: `Board actions for ${sourceProjectName}` }).click();
  await exportPage.getByRole('menuitem', { exact: true, name: 'Export project (.invk)' }).waitFor();
  await exportPage.getByRole('menuitem', { name: /^Download Board/ }).waitFor();
  await exportPage.keyboard.press('Escape');

  const downloadPromise = exportPage.waitForEvent('download');

  await exportPage
    .getByRole('button', { exact: true, name: `Switch project. Current project: ${sourceProjectName}` })
    .click();
  await exportPage.getByRole('menuitem', { exact: true, name: 'Export' }).click();
  await exportPage.getByRole('dialog', { name: `Export ${sourceProjectName}`, exact: true }).waitFor();
  assert.equal(await exportPage.getByRole('checkbox', { name: 'Include font files', exact: true }).isChecked(), false);
  await exportPage.getByRole('button', { name: 'Export project', exact: true }).click();

  const download = await downloadPromise;
  const archivePath = join(tempDirectory, 'fixture-project-002.invk');

  assert.equal(download.suggestedFilename(), `${sourceProjectName}.invk`);
  await download.saveAs(archivePath);
  await exportContext.close();
  contexts.delete(exportContext);
  assertNoBrowserErrors(errors);

  const entries = await readArchiveEntries(archivePath);
  const manifest = readEntryJson(entries, 'manifest.json');
  const archivedBoard = readEntryJson(entries, 'board.json');
  const bundledImages = Object.keys(entries).filter((name) => name.startsWith('images/'));
  const bundledVideos = Object.keys(entries).filter((name) => name.startsWith('videos/'));

  assert.equal(manifest.version, 2);
  assert.equal(manifest.contents, 'workbench-project');
  assert.equal(archivedBoard.version, 1);
  assert.deepEqual(boardShape(archivedBoard.items), EXPECTED_BOARD_SHAPE);
  assert.deepEqual(archivedBoard.items.map((item) => item.name).sort(), [...ARCHIVED_BOARD_NAMES].sort());
  // Expect five board items plus three external images and one external video, deduplicated across references.
  assert.deepEqual(
    bundledImages.sort(),
    [...ARCHIVED_BOARD_NAMES.filter((name) => name.endsWith('.png')), ...PROJECT_FILE_BOARD.externalImages]
      .map((name) => `images/${name}`)
      .sort()
  );
  assert.deepEqual(
    bundledVideos.sort(),
    [`videos/${PROJECT_FILE_BOARD.video}`, `videos/${MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME}`].sort()
  );
  // Hidden from every gallery view and from the archive with it.
  assert.equal(entries[`images/${PROJECT_FILE_BOARD.intermediateImage}`], undefined);
  assert.equal(entries[`images/${PROJECT_FILE_BOARD.canvasOwnedImage}`], undefined);

  // Preserve two collisions: board-owned media must be copied; an external reference must be reused.
  const collide = [PROJECT_FILE_BOARD.referencedImage, PROJECT_FILE_BOARD.video, PROJECT_FILE_BOARD.externalImages[0]];
  const reset = await fetchJson(`/__reset?profile=empty&collide=${collide.map(encodeURIComponent).join(',')}`, {
    method: 'POST',
  });

  assert.equal(reset.profile, 'empty');
  // Nothing survives the reset except the deliberate collisions, and the profile says so.
  assert.deepEqual(reset.counts, { ...MOCK_BACKEND_PROFILE_COUNTS.empty, images: 2 });
  assert.equal(backend.profile(), 'empty');

  const { context: importContext, page: importPage } = await importArchive({
    archivePath,
    browser,
    contexts,
    errors,
    phase: 'import',
  });

  const projects = await fetchJson('/api/v1/projects/');

  assert.equal(projects.length, 1);
  const [summary] = projects;

  assert.ok(summary);
  assert.notEqual(summary.project_id, sourceProjectId);
  // The project's board is the server's to mint; the archived one meant nothing here.
  assert.notEqual(summary.board_id, PROJECT_FILE_BOARD_ID);
  await importPage.getByRole('main', { exact: true, name: summary.name }).waitFor();
  assert.equal(new URL(importPage.url()).hash, '#/app');

  const imported = await fetchJson(`/api/v1/projects/${encodeURIComponent(summary.project_id)}`);
  const importedBoard = await getBoardSnapshot(summary.project_id);
  const importedBoardNames = importedBoard.items.map((item) => item.name);

  assert.equal(imported.data.id, summary.project_id);
  assert.deepEqual(boardShape(importedBoard.items), EXPECTED_BOARD_SHAPE);

  for (const name of importedBoardNames) {
    assert.equal(ARCHIVED_BOARD_NAMES.includes(name), false, `${name} was adopted from the archive instead of copied.`);
  }

  const importedLayers = getLayerImageNames(imported);
  const [firstExternal, ...otherExternals] = PROJECT_FILE_BOARD.externalImages;

  assert.equal(importedLayers.filter((name) => importedBoardNames.includes(name)).length, 1);
  assert.equal(importedLayers.includes(PROJECT_FILE_BOARD.referencedImage), false);
  assert.equal(importedLayers.includes(firstExternal), true, 'an existing identity must satisfy a reference');

  for (const name of otherExternals) {
    assert.equal(importedLayers.includes(name), false, `${name} should have been uploaded under a new name`);
  }

  const importedVideoName = getDocumentVideoName(imported);
  const importedVideos = await fetchJson('/api/v1/videos/?limit=100&offset=0');
  const importedVideoNames = new Set(importedVideos.items.map((video) => video.video_name));

  assert.notEqual(importedVideoName, MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME);
  assert.equal(importedVideoNames.has(importedVideoName), true);

  // Adopting the collision would move it onto the restored board; assert it stays unboarded.
  const collidedVideo = importedVideos.items.find((video) => video.video_name === PROJECT_FILE_BOARD.video);
  const collidedImage = await fetchJson(`/api/v1/images/i/${encodeURIComponent(PROJECT_FILE_BOARD.referencedImage)}`);

  assert.ok(collidedVideo, 'the video collision fixture must survive the reset');
  assert.equal(collidedVideo.board_id, null);
  assert.equal(collidedImage.board_id, null);

  // Nothing the source hid came across.
  for (const hidden of [PROJECT_FILE_BOARD.intermediateImage, PROJECT_FILE_BOARD.canvasOwnedImage]) {
    const response = await fetch(`${backendOrigin}/api/v1/images/i/${encodeURIComponent(hidden)}`);

    assert.equal(response.status, 404, `${hidden} must not exist on the destination.`);
  }

  for (const item of importedBoard.items) {
    const base = item.kind === 'image' ? '/api/v1/images' : '/api/v1/videos';

    await assertAssetRoute(base, item.name, ['full', 'thumbnail']);
  }

  const restoredImageNames = new Set(
    (await fetchJson('/api/v1/images/?limit=200&offset=0')).items.map((image) => image.image_name)
  );
  const coverImageName = await waitForProjectCover(summary.project_id, restoredImageNames);

  await importContext.close();
  contexts.delete(importContext);
  assertNoBrowserErrors(errors);

  // The fixture was a schema-2 document with one `projectGraph`; import canonicalizes it into the current
  // schema with that graph as the only workflow and no library write target.
  assert.equal(imported.data.documentSchemaVersion, 3);
  assert.equal(imported.data.workflows.entries.length, 1);
  assert.equal(imported.data.workflows.activeWorkflowId, imported.data.workflows.entries[0].document.id);
  assert.equal(imported.data.workflows.entries[0].source, undefined);

  await runDuplication({ browser, contexts, errors, imported, importedBoardNames });
  await runWorkflowCollection({ browser, contexts, errors, imported });
  await runMissingBinaryImport({ browser, contexts, entries, errors, tempDirectory });

  return {
    boardItemCount: importedBoard.items.length,
    coverImageName,
    imageNames: [...importedBoardNames].sort(),
    projectId: summary.project_id,
    videoName: importedVideoName,
  };
};

/** Duplication copies board media but reuses external references on the same server. */
const runDuplication = async ({ browser, contexts, errors, imported, importedBoardNames }) => {
  const context = await browser.newContext();

  contexts.add(context);
  const page = await context.newPage();

  observeBrowserErrors(page, 'duplicate', errors);
  // Enter through Home so authentication settles before opening the library.
  await page.goto(`${origin}/#/`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { exact: true, name: 'Welcome to Invoke' }).waitFor();
  await page.goto(`${origin}/#/projects`, { waitUntil: 'domcontentloaded' });

  const card = page.getByRole('link', { exact: true, name: `Open ${sourceProjectName}` }).first();

  await card.waitFor();
  await card.hover();
  await page.getByRole('button', { exact: true, name: 'Actions' }).first().click();
  await page.getByRole('menuitem', { exact: true, name: 'Duplicate' }).click();
  await page.getByText('Project duplicated').waitFor();

  const projects = await fetchJson('/api/v1/projects/');

  assert.equal(projects.length, 2);

  const copy = projects.find((project) => project.project_id !== imported.project_id);

  assert.ok(copy, 'The duplication did not create a second project.');
  assert.equal(copy.name, `${sourceProjectName} copy`);
  assert.notEqual(copy.board_id, imported.board_id);

  const copiedBoard = await getBoardSnapshot(copy.project_id);
  const copiedBoardNames = copiedBoard.items.map((item) => item.name);

  assert.deepEqual(boardShape(copiedBoard.items), EXPECTED_BOARD_SHAPE);

  for (const name of copiedBoardNames) {
    assert.equal(importedBoardNames.includes(name), false, `${name} is shared with the project it was copied from.`);
  }

  const copiedRecord = await fetchJson(`/api/v1/projects/${encodeURIComponent(copy.project_id)}`);
  const copiedLayers = getLayerImageNames(copiedRecord);
  const importedLayers = getLayerImageNames(imported);

  assert.equal(copiedLayers.filter((name) => copiedBoardNames.includes(name)).length, 1);
  // Everything outside the board is a pointer, and the pointer is still good.
  assert.deepEqual(
    copiedLayers.filter((name) => !copiedBoardNames.includes(name)).sort(),
    importedLayers.filter((name) => !importedBoardNames.includes(name)).sort()
  );
  assert.equal(getDocumentVideoName(copiedRecord), getDocumentVideoName(imported));

  await context.close();
  contexts.delete(context);
  assertNoBrowserErrors(errors);
};

/**
 * The primary workflow journey: open a template into the project, edit it, switch away and back, save it to the
 * library under a new name, restart, and find both workflows intact. Project edits and runs issue no library writes;
 * the This-project view fetches no templates.
 */
const runWorkflowCollection = async ({ browser, contexts, errors, imported }) => {
  const projectId = imported.project_id;
  const template = await fetchJson('/api/v1/workflows/', {
    body: JSON.stringify({
      workflow: {
        author: '',
        contact: '',
        description: 'A template the journey opens into the project.',
        edges: [],
        exposedFields: [],
        form: {
          elements: { root: { data: { children: [], layout: 'column' }, id: 'root', type: 'container' } },
          rootElementId: 'root',
        },
        meta: { category: 'user', version: '3.0.0' },
        name: 'Journey Template',
        nodes: [],
        notes: '',
        tags: 'journey',
        version: '1.0.0',
      },
    }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });
  const requestsBefore = (await workflowRequests()).length;

  const context = await browser.newContext();

  contexts.add(context);
  const page = await context.newPage();

  observeBrowserErrors(page, 'workflow-collection', errors);
  await page.goto(`${origin}/#/`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { exact: true, name: 'Welcome to Invoke' }).waitFor();
  await page.goto(`${origin}/#/app?project=${encodeURIComponent(projectId)}`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('main', { exact: true, name: imported.name }).waitFor();
  await selectLayoutPreset(page, 'Automate', 'Workflow');

  const workflowName = () => page.getByRole('button', { name: /^Open this project's workflows\. Current workflow: / });
  const dialog = page.getByRole('dialog', { exact: true, name: 'Workflows' });
  const cards = dialog.locator('[data-workflow-card]');
  const openProjectWorkflows = async () => {
    await workflowName().click();
    await dialog.waitFor();
    await dialog.locator('[data-library-tab="project"]').waitFor();
  };
  const currentWorkflowName = async () =>
    (await workflowName().getAttribute('aria-label')).replace("Open this project's workflows. Current workflow: ", '');

  // The migrated project owns exactly its old graph, marked active; opening the local view fetched nothing.
  await openProjectWorkflows();
  assert.equal(await cards.count(), 1);
  await cards.first().locator('[data-active-workflow]').waitFor();
  assert.equal(
    (await workflowRequests()).slice(requestsBefore).some((entry) => entry.method === 'GET'),
    false
  );

  // A template opened from the library tabs becomes an independent copy in this project.
  await dialog.getByText('Yours', { exact: true }).click();
  await dialog.locator('[data-library-tab="user"]').waitFor();
  await dialog.locator(`[data-workflow-card="${template.workflow_id}"]`).click();
  await dialog
    .locator(`[data-workflow-detail="${template.workflow_id}"]`)
    .getByRole('button', { exact: true, name: 'Open' })
    .click();
  await dialog.waitFor({ state: 'hidden' });
  assert.equal(await currentWorkflowName(), 'Journey Template');

  // Editing the copy: rename it through the local view. The library is not written.
  await openProjectWorkflows();
  assert.equal(await cards.count(), 2);
  // The rail's menu, not a tile's: every tile now carries its own More-actions button.
  await dialog
    .locator('[data-project-workflow-detail]')
    .getByRole('button', { exact: true, name: 'More actions' })
    .click();
  await page.getByRole('menuitem', { name: /^Rename…/ }).click();
  const renameDialog = page.getByRole('dialog', { exact: true, name: 'Rename workflow' });
  await renameDialog.waitFor();
  await renameDialog.getByRole('textbox').fill('Journey Template edited');
  await renameDialog.getByRole('button', { exact: true, name: 'Rename' }).click();
  await renameDialog.waitFor({ state: 'hidden' });
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'hidden' });
  assert.equal(await currentWorkflowName(), 'Journey Template edited');

  // Switching to the other workflow and back keeps both documents.
  await openProjectWorkflows();
  await cards.filter({ hasText: 'Empty Workflow' }).dblclick();
  await dialog.waitFor({ state: 'hidden' });
  assert.equal(await currentWorkflowName(), 'Empty Workflow');
  await openProjectWorkflows();
  await cards.filter({ hasText: 'Journey Template edited' }).dblclick();
  await dialog.waitFor({ state: 'hidden' });
  assert.equal(await currentWorkflowName(), 'Journey Template edited');

  const libraryWritesSoFar = (await workflowRequests())
    .slice(requestsBefore)
    .filter((entry) => entry.method === 'PATCH' || entry.method === 'POST' || entry.method === 'DELETE');

  assert.deepEqual(libraryWritesSoFar, [], 'project edits must not write to the workflow library');

  // Save the edited copy under a new library name, without duplicating it beforehand.
  await page.getByRole('button', { exact: true, name: 'Save to library…' }).click();
  const saveDialog = page.getByRole('dialog', { exact: true, name: 'Save to library' });
  await saveDialog.waitFor();
  await saveDialog.getByRole('textbox').fill('Journey Saved');
  await saveDialog.getByRole('button', { exact: true, name: 'Save' }).click();
  await page.getByText('Workflow saved', { exact: true }).waitFor();

  const libraryWrites = (await workflowRequests())
    .slice(requestsBefore)
    .filter((entry) => entry.method === 'PATCH' || entry.method === 'POST' || entry.method === 'DELETE');

  assert.deepEqual(libraryWrites, [{ method: 'POST', path: '/api/v1/workflows/' }]);

  const library = await fetchJson('/api/v1/workflows/?categories=user&page=0&per_page=50');
  const saved = library.items.find((item) => item.name === 'Journey Saved');

  assert.ok(saved, 'the save created a new template');
  assert.equal(saved.revision, 1);
  assert.equal(library.items.find((item) => item.workflow_id === template.workflow_id)?.name, 'Journey Template');

  // The project keeps both workflows; the saved copy now targets the new template, and the name stayed the copy's.
  const record = await waitForSavedWorkflows(
    projectId,
    (candidate) =>
      candidate.data.workflows?.entries?.length === 2 &&
      candidate.data.workflows.entries[1].source?.libraryWorkflowId === saved.workflow_id,
    'two workflows with the saved copy linked to its new template'
  );

  assert.equal(record.data.documentSchemaVersion, 3);
  assert.deepEqual(
    record.data.workflows.entries.map((entry) => entry.document.name),
    ['Empty Workflow', 'Journey Template edited']
  );
  assert.deepEqual(record.data.workflows.entries[1].source, { libraryWorkflowId: saved.workflow_id, revision: 1 });
  assert.equal(record.data.workflows.activeWorkflowId, record.data.workflows.entries[1].document.id);

  // Restart: both workflows and the active selection come back from the server.
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByRole('main', { exact: true, name: imported.name }).waitFor();
  await selectLayoutPreset(page, 'Automate', 'Workflow');
  assert.equal(await currentWorkflowName(), 'Journey Template edited');
  await openProjectWorkflows();
  assert.equal(await cards.count(), 2);
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'hidden' });

  await context.close();
  contexts.delete(context);
  assertNoBrowserErrors(errors);
};

/** Count missing board media separately from missing document references. */
const runMissingBinaryImport = async ({ browser, contexts, entries, errors, tempDirectory }) => {
  const damagedPath = await writeArchiveWithout(
    entries,
    [
      `images/${PROJECT_FILE_BOARD.unreferencedImage}`,
      `images/${PROJECT_FILE_BOARD.userAsset}`,
      `images/${PROJECT_FILE_BOARD.externalImages[1]}`,
    ],
    join(tempDirectory, 'fixture-project-002-damaged.invk')
  );
  const { context, page } = await importArchive({
    archivePath: damagedPath,
    browser,
    contexts,
    errors,
    phase: 'damaged-import',
  });

  await page.getByText('2 board items could not be included.').waitFor();
  await page.getByText('1 project reference could not be included.').waitFor();

  await context.close();
  contexts.delete(context);
  assertNoBrowserErrors(errors);
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
  killPreview,
  launchBrowser: ({ timeoutMs }) => chromium.launch({ headless: true, timeout: timeoutMs }),
  now: () => performance.now(),
  removeTempDirectory: (directory) => rm(directory, { force: true, recursive: true }),
  runRoundTrip,
  spawnPreview: () =>
    spawnPreview({
      cwd: root,
      env: { ...process.env, INVOKEAI_DEV_BACKEND: backendOrigin },
      port,
      stdio: ['ignore', 'ignore', 'pipe'],
    }),
  startBackend: () => startMockBackend(backendPort, { profile: 'representative' }),
  waitForPreview: ({ getPreviewExit }) => waitForPreview(getPreviewExit),
});

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
  const acquire = (label, create, dispose, assign) => {
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
      dependencies.killPreview(preview.pid, 'SIGTERM');
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
      dependencies.killPreview(preview.pid, 'SIGKILL');
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
