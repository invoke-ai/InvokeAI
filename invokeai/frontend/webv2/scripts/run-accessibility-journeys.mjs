import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright';

import { assertNoAxeViolations } from './accessibility/axe.mjs';
import { startMockBackend } from './mock-backend.mjs';

const root = resolve(import.meta.dirname, '..');
const port = Number(process.env.INVOKEAI_ACCESSIBILITY_PORT ?? 4178);
const origin = `http://127.0.0.1:${String(port)}`;
const backendPort = Number(process.env.INVOKEAI_ACCESSIBILITY_BACKEND_PORT ?? 4179);
const backendOrigin = `http://127.0.0.1:${String(backendPort)}`;
const representativeProjectPath = '/#/app?project=fixture-project-001';

const waitForPreview = async () => {
  const deadline = Date.now() + 20_000;

  while (Date.now() < deadline) {
    try {
      const response = await fetch(origin);

      if (response.ok) {
        return;
      }
    } catch {
      // Preview is still starting.
    }

    await new Promise((resolveWait) => {
      setTimeout(resolveWait, 100);
    });
  }

  throw new Error(`Vite preview did not become ready at ${origin}.`);
};

const waitForSettledDocument = async (page) => {
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolveFrame) => {
      requestAnimationFrame(() => requestAnimationFrame(resolveFrame));
    });
  });
};

const openRepresentativePage = async (browser, path) => {
  const resetResponse = await fetch(`${backendOrigin}/__reset?profile=representative`, { method: 'POST' });

  if (!resetResponse.ok) {
    throw new Error(`Could not reset representative backend: ${resetResponse.status}.`);
  }

  const context = await browser.newContext({
    colorScheme: 'dark',
    reducedMotion: 'reduce',
    viewport: { height: 1_000, width: 1_440 },
  });
  const page = await context.newPage();
  const pageErrors = [];

  page.on('pageerror', (error) => pageErrors.push(error));
  await page.goto(`${origin}${path}`, { waitUntil: 'domcontentloaded' });

  return { context, page, pageErrors };
};

const waitForProjects = async (page) => {
  await page.getByRole('heading', { exact: true, name: 'Welcome to Invoke' }).waitFor();
  await page.getByRole('link', { exact: true, name: 'Open Fixture Project 001' }).waitFor();
};

const waitForModels = async (page) => {
  await page.getByLabel('Model library', { exact: true }).waitFor();
  await page.getByText('Fixture Model 001', { exact: true }).waitFor();
};

const waitForNodes = async (page) => {
  await page.getByRole('textbox', { exact: true, name: 'Search node packs' }).waitFor();
  await page.getByText('fixture-pack-01', { exact: true }).waitFor();
};

const waitForWorkbench = async (page) => {
  await page.getByRole('main', { exact: true, name: 'Fixture Project 001' }).waitFor();
  await page.getByRole('button', { name: /^Layout preset:/ }).waitFor();
};

const selectLayoutPreset = async (page, preset) => {
  const trigger = page.getByRole('button', { name: /^Layout preset:/ });

  await trigger.click();
  await page.getByRole('menuitem', { exact: true, name: preset }).click();
  await page.getByRole('button', { exact: true, name: `Layout preset: ${preset}` }).waitFor();
  await page.getByRole('tab', { exact: true, name: preset }).waitFor();
};

const surfaces = [
  {
    id: 'launchpad-projects-representative',
    path: '/#/',
    ready: waitForProjects,
  },
  {
    id: 'launchpad-models-representative',
    path: '/#/models',
    ready: waitForModels,
  },
  {
    id: 'launchpad-nodes-representative',
    path: '/#/nodes',
    ready: waitForNodes,
  },
  {
    id: 'workbench-default-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await page.getByRole('tab', { exact: true, name: 'Preview' }).waitFor();
    },
  },
  {
    id: 'workbench-canvas-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Canvas');
    },
  },
  {
    id: 'workbench-gallery-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Gallery');
      await page.getByRole('list', { exact: true, name: 'Gallery images' }).waitFor();
    },
  },
  {
    id: 'workbench-workflow-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Workflow');
      await page.getByText('Fixture Node 001', { exact: true }).waitFor();
    },
  },
];

const runAxeSurface = async (browser, surface) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, surface.path);

  try {
    await surface.ready(page);
    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, surface.id);

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${surface.id} raised uncaught browser errors.`);
    }

    return { id: surface.id, status: 'passed' };
  } finally {
    await context.close();
  }
};

const expectFocused = async (locator, message) => {
  assert.equal(await locator.evaluate((element) => element === document.activeElement), true, message);
};

const runKeyboardJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, '/#/');

  try {
    await waitForProjects(page);

    const projectsTab = page.getByRole('tab', { exact: true, name: 'Projects' });
    const modelsTab = page.getByRole('tab', { exact: true, name: 'Models' });

    await projectsTab.focus();
    await projectsTab.press('ArrowDown');
    await expectFocused(modelsTab, 'ArrowDown should move focus from Projects to Models.');
    await waitForModels(page);
    assert.match(page.url(), /#\/models$/);
    assert.equal(await modelsTab.getAttribute('aria-selected'), 'true');

    const paletteTrigger = page.getByRole('button', { exact: true, name: 'Command palette' });

    await paletteTrigger.click();
    const paletteDialog = page.getByRole('dialog', { exact: true, name: 'Command palette' });
    const paletteInput = page.getByRole('combobox', { exact: true, name: 'Search commands and settings' });

    await paletteDialog.waitFor();
    await expectFocused(paletteInput, 'Opening the command palette should focus its search field.');
    await paletteInput.press('Escape');
    await paletteDialog.waitFor({ state: 'hidden' });
    await expectFocused(paletteTrigger, 'Closing the command palette should restore focus to its trigger.');

    await projectsTab.focus();
    await projectsTab.press('Enter');
    await waitForProjects(page);

    const projectLink = page.getByRole('link', { exact: true, name: 'Open Fixture Project 001' });

    await projectLink.focus();
    await projectLink.press('Enter');
    await waitForWorkbench(page);

    const previewTab = page.getByRole('tab', { exact: true, name: 'Preview' });
    const canvasTab = page.getByRole('tab', { exact: true, name: 'Canvas' });

    await previewTab.focus();
    await previewTab.press('ArrowRight');
    await expectFocused(canvasTab, 'ArrowRight should move focus to the next center view.');
    assert.equal(await canvasTab.getAttribute('aria-selected'), 'true');

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, 'keyboard-critical-journey raised uncaught browser errors.');
    }

    return { id: 'keyboard-critical-journey', status: 'passed' };
  } finally {
    await context.close();
  }
};

const mockBackend = await startMockBackend(backendPort, { profile: 'representative' });
const preview = spawn(
  'pnpm',
  ['exec', 'vite', 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  {
    cwd: root,
    detached: true,
    env: { ...process.env, INVOKEAI_DEV_BACKEND: backendOrigin },
    stdio: ['ignore', 'pipe', 'pipe'],
  }
);
let previewError = '';
let browser = null;

preview.stderr.on('data', (chunk) => {
  previewError += String(chunk);
});

try {
  await waitForPreview();
  browser = await chromium.launch({ headless: true });
  const reports = [];

  for (const surface of surfaces) {
    reports.push(await runAxeSurface(browser, surface));
  }

  reports.push(await runKeyboardJourney(browser));
  process.stdout.write(`${JSON.stringify({ profile: 'representative', reports }, null, 2)}\n`);
} catch (error) {
  throw new Error(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}${previewError ? `\n${previewError}` : ''}`
  );
} finally {
  await browser?.close();

  if (preview.pid) {
    try {
      process.kill(-preview.pid, 'SIGTERM');
    } catch {
      // Preview may already have exited after a startup failure.
    }
  }

  await mockBackend.close();
}
