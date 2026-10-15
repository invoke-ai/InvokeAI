import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright';

import { assertNoAxeViolations } from './accessibility/axe.mjs';
import { MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME } from './mock-backend-fixtures.mjs';
import { startMockBackend } from './mock-backend.mjs';
import { killPreview, spawnPreview } from './preview-server.mjs';

const root = resolve(import.meta.dirname, '..');
const port = Number(process.env.INVOKEAI_ACCESSIBILITY_PORT ?? 4178);
const origin = `http://127.0.0.1:${String(port)}`;
const backendPort = Number(process.env.INVOKEAI_ACCESSIBILITY_BACKEND_PORT ?? 4179);
const backendOrigin = `http://127.0.0.1:${String(backendPort)}`;
const representativeProjectPath = '/#/app?project=fixture-project-001';
const requestedJourney = process.env.INVOKEAI_ACCESSIBILITY_JOURNEY;

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
    // Wait for animations to finish before contrast audits. Keep this serialized helper aligned with
    // settleAnimations.testing.ts.
    await Promise.allSettled(
      document
        .getAnimations({ subtree: true })
        .filter((animation) => {
          const timing = animation.effect?.getComputedTiming();
          const duration = typeof timing?.duration === 'number' ? timing.duration : 0;

          // A spinner, a shimmering skeleton, or a paused animation never finishes.
          return (
            timing?.iterations !== Infinity &&
            Number.isFinite(duration) &&
            animation.playbackRate !== 0 &&
            animation.playState !== 'paused'
          );
        })
        .map((animation) => animation.finished)
    );
    await new Promise((resolveFrame) => {
      requestAnimationFrame(() => requestAnimationFrame(resolveFrame));
    });
  });
};

const openRepresentativePage = async (browser, path, viewport = { height: 1_000, width: 1_440 }, themeId) => {
  const resetResponse = await fetch(`${backendOrigin}/__reset?profile=representative`, { method: 'POST' });

  if (!resetResponse.ok) {
    throw new Error(`Could not reset representative backend: ${resetResponse.status}.`);
  }

  if (themeId) {
    const response = await fetch(
      `${backendOrigin}/api/v1/client_state/default/set_by_key?key=webv2%3Aworkbench-settings`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(
          JSON.stringify({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: 'fixture', themeId })
        ),
      }
    );
    assert.ok(response.ok, `Could not set ${themeId} accessibility fixture theme.`);
  }

  const context = await browser.newContext({
    colorScheme: themeId === 'light' ? 'light' : 'dark',
    reducedMotion: 'reduce',
    viewport,
  });
  const page = await context.newPage();
  const pageErrors = [];
  const consoleErrors = [];

  page.on('pageerror', (error) => pageErrors.push(error));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text());
    }
  });
  await page.goto(`${origin}${path}`, { waitUntil: 'domcontentloaded' });

  return { consoleErrors, context, page, pageErrors };
};

/** `/` is Home: the greeting, the resume card, and the intent tiles. */
const waitForHome = async (page) => {
  await page.getByRole('heading', { exact: true, name: 'Welcome to Invoke' }).waitFor();
  await page.getByRole('link', { exact: true, name: 'Open Fixture Project 001' }).waitFor();
  await page.getByText('Generate from text', { exact: true }).waitFor();
};

/** Use the level-2 library heading; the shell has a hidden level-1 heading with the same name. */
const waitForProjects = async (page) => {
  await page.getByRole('heading', { exact: true, level: 2, name: 'Projects' }).waitFor();
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

const presetStrip = (page) => page.getByRole('tablist', { exact: true, name: 'Layout preset' });

const waitForWorkbench = async (page) => {
  await page.getByRole('main', { exact: true, name: 'Fixture Project 001' }).waitFor();
  await presetStrip(page).waitFor();
};

const centerViewTrigger = (page, label) => page.getByRole('button', { exact: true, name: `Center view: ${label}` });

/** Preset names need not match their center views; pass the expected view explicitly. */
const selectLayoutPreset = async (page, preset, centerView) => {
  // Match the full preset name plus optional drift marker within the strip; custom presets and Workflow tabs can
  // share prefixes.
  const name = new RegExp(`^${preset}(, unsaved changes)?$`);
  const strip = presetStrip(page);
  const selected = strip.getByRole('tab', { name, selected: true });

  if ((await selected.count()) === 0) {
    await strip.getByRole('tab', { name }).click();
  }

  await selected.waitFor();
  await centerViewTrigger(page, centerView).waitFor();
};

/** Compose keeps a Gallery in the right rail too, so gallery locators are scoped. */
const centerRegion = (page) => page.getByRole('region', { exact: true, name: 'Center view' });

const selectCenterView = async (page, from, to) => {
  await centerViewTrigger(page, from).click();
  await page.getByRole('menuitemradio', { exact: true, name: to }).click();
  await centerViewTrigger(page, to).waitFor();
};

const surfaces = [
  {
    id: 'launchpad-home-representative',
    path: '/#/',
    ready: waitForHome,
  },
  {
    id: 'launchpad-projects-representative',
    path: '/#/projects',
    ready: waitForProjects,
  },
  {
    id: 'launchpad-preferences-intermediates-representative',
    path: '/#/projects',
    ready: async (page) => {
      await page
        .getByRole('navigation', { exact: true, name: 'Launchpad sections' })
        .getByRole('link', { exact: true, name: 'Preferences' })
        .click();
      await page
        .getByRole('list', { exact: true, name: 'Settings section' })
        .getByRole('button', { exact: true, name: 'Intermediates' })
        .click();
      await page.getByRole('heading', { exact: true, level: 2, name: 'Intermediates' }).waitFor();
      await page.getByRole('textbox', { exact: true, name: 'Search projects' }).waitFor();
      await page.getByRole('checkbox', { exact: true, name: 'Select Fixture Project 001' }).waitFor();
    },
    representative: true,
  },
  {
    id: 'launchpad-fonts-empty',
    path: '/#/fonts',
    ready: async (page) => {
      await page.getByRole('textbox', { exact: true, name: 'Search fonts' }).waitFor();
      await page.getByText('No fonts available', { exact: true }).waitFor();
    },
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
      await centerViewTrigger(page, 'Preview').waitFor();
    },
  },
  {
    id: 'workbench-canvas-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Edit', 'Canvas');
      // The Layers panel's Properties pane is the canvas's settings surface; the journey must scan it, not a fallback.
      await page.getByRole('tab', { exact: true, name: 'Properties', selected: true }).waitFor();
      await page.getByRole('tabpanel').getByRole('group', { exact: true, name: 'Layer' }).waitFor();
    },
  },
  {
    id: 'workbench-gallery-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Compose', 'Preview');
      await selectCenterView(page, 'Preview', 'Gallery');
      // Choose the populated board row; the disclosure also names the currently selected board.
      await centerRegion(page).locator('button:not([aria-expanded])').filter({ hasText: 'Uncategorized' }).click();
      await centerRegion(page).getByRole('list', { exact: true, name: 'Gallery items' }).waitFor();
    },
  },
  {
    id: 'workbench-workflow-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Automate', 'Workflow');
      await page.getByText('Fixture Node 001', { exact: true }).waitFor();
      // Pan the y=0 fixture clear of the floating toolbar before measuring targets.
      const pane = await page.locator('.react-flow__pane').boundingBox();
      assert.ok(pane);
      const x = pane.x + pane.width / 2;
      const y = pane.y + 160;
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x, y + 64, { steps: 4 });
      await page.mouse.up();
      await page
        .locator('[data-id="fixture-workflow-node-001"]')
        .getByRole('button', { name: 'Collapse node' })
        .click({ trial: true });
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
  } catch (error) {
    const artifactDir = resolve(root, 'artifacts/accessibility');
    await mkdir(artifactDir, { recursive: true })
      .then(() => page.screenshot({ path: resolve(artifactDir, `${surface.id}.png`) }))
      .catch(() => undefined);
    throw error;
  } finally {
    await context.close();
  }
};

/** Poll focus because roving tabindex and focus restoration run after the initiating key event. */
const expectFocused = async (locator, message) => {
  const deadline = Date.now() + 5_000;

  for (;;) {
    if (await locator.evaluate((element) => element === document.activeElement)) {
      return;
    }

    if (Date.now() >= deadline) {
      const activeElement = await locator.page().evaluate(() => {
        const active = document.activeElement;

        return active instanceof HTMLElement
          ? {
              ariaLabel: active.getAttribute('aria-label'),
              role: active.getAttribute('role'),
              tagName: active.tagName,
              text: active.innerText.slice(0, 200),
            }
          : null;
      });
      assert.fail(`${message} Active element: ${JSON.stringify(activeElement)}.`);
    }

    await new Promise((resolveWait) => {
      setTimeout(resolveWait, 25);
    });
  }
};

const runShortcutGuideJourney = async (browser) => {
  for (const themeId of ['classic', 'light']) {
    const { context, page, pageErrors } = await openRepresentativePage(
      browser,
      representativeProjectPath,
      undefined,
      themeId
    );
    try {
      await waitForWorkbench(page);
      await page.waitForFunction((theme) => document.documentElement.dataset.theme === theme, themeId);
      await selectLayoutPreset(page, 'Edit', 'Canvas');
      const viewTool = page
        .getByRole('toolbar', { exact: true, name: 'Tools' })
        .getByRole('button', { exact: true, name: 'View' });
      await viewTool.click();
      const guide = page.locator('footer').getByRole('button', { exact: true, name: 'Shortcuts' });
      await guide.waitFor();
      await waitForSettledDocument(page);
      await assertNoAxeViolations(page, `shortcuts-${themeId}-compact`, { include: ['footer [data-shortcut-guide]'] });
      await guide.focus();
      await guide.press('Enter');
      const popup = page
        .getByRole('dialog')
        .filter({ has: page.getByRole('button', { exact: true, name: 'Keyboard shortcut settings' }) });
      await popup.waitFor();
      await waitForSettledDocument(page);
      await assertNoAxeViolations(page, `shortcuts-${themeId}-expanded`, {
        include: ['[data-scope="popover"][data-part="content"][data-state="open"][data-workbench-focus-preserve]'],
      });
      await popup.getByRole('button', { exact: true, name: 'Keyboard shortcut settings' }).press('Escape');
      await popup.waitFor({ state: 'hidden' });
      await expectFocused(viewTool, 'Closing the guide should restore its editing origin.');
      if (pageErrors.length > 0) {
        throw new AggregateError(pageErrors, `shortcuts-${themeId} raised uncaught browser errors.`);
      }
    } catch (error) {
      const directory = resolve(root, 'artifacts/accessibility');
      await mkdir(directory, { recursive: true });
      await page.screenshot({ path: resolve(directory, `shortcuts-${themeId}.png`) }).catch(() => undefined);
      throw error;
    } finally {
      await context.close();
    }
  }
  return { id: 'workbench-shortcuts', status: 'passed' };
};

const runKeyboardJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, '/#/');

  try {
    await waitForHome(page);

    const rail = page.getByRole('navigation', { exact: true, name: 'Launchpad sections' });
    const projectsLink = rail.getByRole('link', { exact: true, name: 'Projects' });
    const nodesLink = rail.getByRole('link', { exact: true, name: 'Nodes' });
    const modelsLink = rail.getByRole('link', { exact: true, name: 'Models' });

    // The rail's links are in visual order, so Tab and reading order follow what it shows.
    const railLinkTops = await rail
      .getByRole('link')
      .evaluateAll((links) => links.map((link) => link.getBoundingClientRect().top));
    assert.deepEqual(
      railLinkTops,
      [...railLinkTops].sort((a, b) => a - b),
      'Launchpad rail links must follow their visual order.'
    );

    await modelsLink.focus();
    await modelsLink.press('Enter');
    await waitForModels(page);
    assert.match(page.url(), /#\/models$/);
    assert.equal(await modelsLink.getAttribute('aria-current'), 'page');
    await modelsLink.press('Tab');
    await expectFocused(nodesLink, 'Tab should move focus from Models to Nodes.');
    await nodesLink.press('Enter');
    await waitForNodes(page);
    assert.match(page.url(), /#\/nodes$/);
    assert.equal(await nodesLink.getAttribute('aria-current'), 'page');
    assert.equal(await modelsLink.getAttribute('aria-current'), null);

    const paletteTrigger = page.getByRole('button', { exact: true, name: 'Command palette' });

    await paletteTrigger.click();
    const paletteDialog = page.getByRole('dialog', { exact: true, name: 'Command palette' });
    const paletteInput = page.getByRole('combobox', { exact: true, name: 'Search commands and settings' });

    await paletteDialog.waitFor();
    await expectFocused(paletteInput, 'Opening the command palette should focus its search field.');
    await paletteInput.press('Escape');
    await paletteDialog.waitFor({ state: 'hidden' });
    await expectFocused(paletteTrigger, 'Closing the command palette should restore focus to its trigger.');

    await projectsLink.focus();
    await projectsLink.press('Enter');
    await waitForProjects(page);

    const projectLink = page.getByRole('link', { exact: true, name: 'Open Fixture Project 001' });

    await projectLink.focus();
    await projectLink.press('Enter');
    await waitForWorkbench(page);

    const previewTrigger = centerViewTrigger(page, 'Preview');

    await previewTrigger.focus();
    await previewTrigger.press('Enter');

    // Compose keeps Preview and Gallery as its center views; Canvas lives under "Add to center".
    const previewItem = page.getByRole('menuitemradio', { exact: true, name: 'Preview' });
    const galleryItem = page.getByRole('menuitemradio', { exact: true, name: 'Gallery' });
    const centerViewMenu = page.getByRole('menu');

    await galleryItem.waitFor();
    assert.equal(await previewItem.getAttribute('aria-checked'), 'true');
    await expectFocused(centerViewMenu, 'Opening the center view menu should focus its composite.');
    assert.equal(await centerViewMenu.getAttribute('aria-activedescendant'), await previewItem.getAttribute('id'));
    await centerViewMenu.press('ArrowDown');
    assert.equal(await centerViewMenu.getAttribute('aria-activedescendant'), await galleryItem.getAttribute('id'));
    await centerViewMenu.press('Enter');
    const galleryTrigger = centerViewTrigger(page, 'Gallery');
    await galleryTrigger.waitFor();
    await expectFocused(galleryTrigger, 'Selecting a center view should restore focus to the view selector.');

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, 'keyboard-critical-journey raised uncaught browser errors.');
    }

    return { id: 'keyboard-critical-journey', status: 'passed' };
  } finally {
    await context.close();
  }
};

const runResponsiveTopbarJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath, {
    height: 900,
    width: 1_024,
  });
  const id = 'workbench-topbar-responsive';

  try {
    await waitForWorkbench(page);
    await page.getByRole('button', { exact: true, name: '0 images remaining. Open queue' }).waitFor();

    for (const width of [1_024, 900]) {
      await page.setViewportSize({ height: 900, width });
      await waitForSettledDocument(page);

      const metrics = await page.locator('header').evaluate((header) => {
        const bounds = header.getBoundingClientRect();
        const zones = [...header.children].map((element) => element.getBoundingClientRect());
        const centerZone = zones[1];

        return {
          centerOffset: Math.abs(bounds.left + bounds.width / 2 - (centerZone.left + centerZone.width / 2)),
          clientWidth: header.clientWidth,
          controlsInsideHeader: [...header.querySelectorAll('button')].every((button) => {
            const controlBounds = button.getBoundingClientRect();

            return controlBounds.left >= bounds.left && controlBounds.right <= bounds.right;
          }),
          scrollWidth: header.scrollWidth,
          zonesDoNotOverlap: zones.every((zone, index) => index === 0 || zones[index - 1].right <= zone.left),
        };
      });

      assert.equal(metrics.scrollWidth, metrics.clientWidth, `The topbar must not overflow at ${width}px.`);
      const documentHeights = await page.evaluate(() => ({
        clientHeight: document.documentElement.clientHeight,
        scrollHeight: document.documentElement.scrollHeight,
      }));
      assert.equal(
        documentHeights.scrollHeight,
        documentHeights.clientHeight,
        `The workbench must not scroll the document at ${width}px.`
      );
      assert.equal(metrics.controlsInsideHeader, true, `Every topbar control must remain visible at ${width}px.`);
      assert.equal(metrics.zonesDoNotOverlap, true, `Topbar zones must not overlap at ${width}px.`);
      assert.ok(
        metrics.centerOffset <= 0.5,
        `The preset strip must be centered at ${width}px (offset: ${metrics.centerOffset.toFixed(2)}px).`
      );
    }

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }

    return { id, status: 'passed' };
  } finally {
    await context.close();
  }
};

const runTopbarMenuJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath);
  const id = 'workbench-topbar-menus';

  try {
    await waitForWorkbench(page);

    // Open Workflow first: the fixture routes from it, but Compose does not mount it by default.
    await centerViewTrigger(page, 'Preview').click();
    await page.getByRole('menuitem', { exact: true, name: 'Workflow' }).click();
    await centerViewTrigger(page, 'Workflow').waitFor();
    await selectCenterView(page, 'Workflow', 'Preview');

    const leftWidgetRail = page.getByRole('navigation', { exact: true, name: 'Create widget visibility' });
    const upscaleWidget = leftWidgetRail.getByRole('button', { exact: true, name: 'Upscale' });
    await upscaleWidget.click({ button: 'right' });
    await page.getByRole('menuitem', { exact: true, name: 'Remove Upscale' }).click();
    assert.equal(await upscaleWidget.count(), 0);
    const activeLeftWidget = leftWidgetRail.getByRole('button', { pressed: true });
    const activeLeftWidgetName = await activeLeftWidget.getAttribute('aria-label');
    assert.ok(activeLeftWidgetName);

    const routingTrigger = page.getByRole('button', { name: /^Invoke from/ });
    const routingLockIndicator = routingTrigger.locator('[data-routing-lock-indicator]');
    const routingTriggerBoundsBeforeHover = await routingTrigger.boundingBox();
    assert.ok(routingTriggerBoundsBeforeHover);
    assert.ok(
      routingTriggerBoundsBeforeHover.width <= 36 && routingTriggerBoundsBeforeHover.height <= 38,
      'The routing trigger should remain narrow and align with its attached controls.'
    );
    assert.equal(await routingLockIndicator.count(), 0);
    await routingTrigger.hover();
    const routingTooltip = page.getByRole('tooltip', { exact: true, name: 'Change routing' });
    await routingTooltip.waitFor({ timeout: 2_000 });
    const [routingTriggerBounds, routingTooltipBounds] = await Promise.all([
      routingTrigger.boundingBox(),
      routingTooltip.boundingBox(),
    ]);
    assert.ok(routingTriggerBounds);
    assert.ok(routingTooltipBounds);
    assert.ok(
      routingTooltipBounds.x < routingTriggerBounds.x + routingTriggerBounds.width &&
        routingTooltipBounds.x + routingTooltipBounds.width > routingTriggerBounds.x,
      'The routing tooltip should be anchored to and horizontally overlap its trigger.'
    );
    await routingTrigger.click();
    const routingMenu = page.getByRole('menu');
    const sourceHeading = routingMenu.getByText('Source', { exact: true });
    const destinationHeading = routingMenu.getByText('Destination', { exact: true });
    const lockRouting = routingMenu.getByRole('menuitem', { exact: true, name: 'Lock routing' });
    await sourceHeading.waitFor();
    await destinationHeading.waitFor();
    await lockRouting.waitFor();
    assert.equal(await routingMenu.getByRole('button', { name: /^(?:Lock|Unlock)/ }).count(), 0);
    assert.equal(await routingMenu.getByText('Opens', { exact: true }).count(), 0);
    assert.equal(await routingMenu.getByText('Selected', { exact: true }).count(), 0);

    await lockRouting.click();
    const lockedRoutingTrigger = page.getByRole('button', {
      exact: true,
      name: 'Invoke from workflow, output to gallery, source locked, destination locked',
    });
    await lockedRoutingTrigger.waitFor();
    assert.equal(await lockedRoutingTrigger.locator('[data-routing-lock-indicator]').count(), 1);
    await lockedRoutingTrigger.click();
    const unlockRouting = page.getByRole('menu').getByRole('menuitem', { exact: true, name: 'Unlock routing' });
    await unlockRouting.click();
    await page
      .getByRole('button', {
        exact: true,
        name: 'Invoke from workflow, output to gallery, following edits',
      })
      .waitFor();
    assert.equal(await routingTrigger.locator('[data-routing-lock-indicator]').count(), 0);

    await routingTrigger.click();

    await routingMenu.getByRole('menuitemradio', { name: /^Upscale/ }).click();
    await page.getByRole('button', { name: /^No source widget open/ }).waitFor();
    await centerViewTrigger(page, 'Preview').waitFor();
    assert.equal(await upscaleWidget.count(), 0);
    assert.equal(await activeLeftWidget.getAttribute('aria-label'), activeLeftWidgetName);

    await page.getByRole('button', { exact: true, name: 'Open menu' }).click();
    const appMenu = page.getByRole('menu', { exact: true, name: 'Open menu' });
    const commandPaletteItem = page.getByRole('menuitem', { name: /^Command palette/ });
    const settingsItem = appMenu.getByRole('menuitem', { name: /^Settings(?: \(.+\))?$/ });
    const whatsNewItem = page.getByRole('menuitem', { exact: true, name: "What's New in Invoke" });
    const documentationItem = page.getByRole('menuitem', { exact: true, name: 'Documentation' });
    const discordItem = page.getByRole('menuitem', { exact: true, name: 'Discord' });
    const donationItem = appMenu.getByRole('menuitem', { exact: true, name: 'Donate to InvokeAI' });
    await commandPaletteItem.waitFor();
    await settingsItem.waitFor();
    await whatsNewItem.waitFor();
    await documentationItem.waitFor();
    await discordItem.waitFor();
    await donationItem.waitFor();
    assert.equal(await donationItem.getAttribute('href'), 'https://github.com/sponsors/invoke-ai');
    assert.equal(await donationItem.getAttribute('target'), '_blank');

    const footerMetrics = await appMenu.evaluate((menu) => {
      const menuBounds = menu.getBoundingClientRect();
      const footerValues = ['command-palette', 'settings', 'whats-new', 'documentation', 'discord'];
      const footerItems = [...menu.querySelectorAll('[role="menuitem"]')].filter((item) =>
        footerValues.includes(item.getAttribute('data-value'))
      );

      return {
        itemsFit: footerItems.every((item) => {
          const bounds = item.getBoundingClientRect();

          return bounds.left >= menuBounds.left && bounds.right <= menuBounds.right && bounds.width <= 32.5;
        }),
        menuClientWidth: menu.clientWidth,
        menuScrollWidth: menu.scrollWidth,
      };
    });
    assert.equal(footerMetrics.itemsFit, true);
    assert.equal(footerMetrics.menuScrollWidth, footerMetrics.menuClientWidth);

    await expectFocused(appMenu, 'Opening the app menu should focus its composite.');
    assert.notEqual(await appMenu.getAttribute('aria-activedescendant'), await commandPaletteItem.getAttribute('id'));
    await new Promise((resolveWait) => {
      setTimeout(resolveWait, 750);
    });
    assert.equal(await page.getByRole('tooltip', { name: /^Command palette/ }).count(), 0);

    await appMenu.press('End');
    assert.equal(await appMenu.getAttribute('aria-activedescendant'), await donationItem.getAttribute('id'));
    await appMenu.press('ArrowUp');
    assert.equal(await appMenu.getAttribute('aria-activedescendant'), await discordItem.getAttribute('id'));
    for (const item of [documentationItem, whatsNewItem, settingsItem, commandPaletteItem]) {
      await appMenu.press('ArrowUp');
      assert.equal(await appMenu.getAttribute('aria-activedescendant'), await item.getAttribute('id'));
    }
    await page.keyboard.press('Escape');

    const projectSwitcher = page.getByRole('button', { name: /^Switch project\./ });
    await projectSwitcher.click();
    await page.getByRole('menuitem', { exact: true, name: 'New project' }).click();
    await page.getByRole('main', { name: /^Project Name #\d+$/ }).waitFor();

    await projectSwitcher.click();
    const openProjects = page.getByRole('menuitemradio');
    assert.ok((await openProjects.count()) >= 2);
    assert.equal(await page.getByRole('menuitemradio', { checked: true }).count(), 1);

    const textOffsets = await openProjects.evaluateAll((items) =>
      items.map((item) => item.querySelector('[data-part="item-text"]')?.getBoundingClientRect().left ?? null)
    );
    assert.equal(
      textOffsets.every((offset) => offset !== null),
      true
    );
    assert.ok(Math.max(...textOffsets) - Math.min(...textOffsets) <= 0.5);

    const otherProject = page.getByRole('menuitemradio', { checked: false }).first();
    const otherProjectName = (await otherProject.locator('[data-part="item-text"]').textContent())?.trim();
    assert.ok(otherProjectName);
    await otherProject.click();
    await page.getByRole('main', { exact: true, name: otherProjectName }).waitFor();

    const customPresetNames = ['Custom One', 'Custom Two', 'Custom Three', 'Custom Four', 'Custom Five', 'Custom Six'];
    for (const presetName of customPresetNames) {
      await page.getByRole('button', { exact: true, name: 'Save this layout as a new preset' }).click();
      const savePresetDialog = page.getByRole('dialog', { exact: true, name: 'Save as new preset' });
      await savePresetDialog.getByRole('textbox', { exact: true, name: 'Preset name' }).fill(presetName);
      await savePresetDialog.getByRole('button', { exact: true, name: 'Save preset' }).click();
      await savePresetDialog.waitFor({ state: 'hidden' });
    }
    for (const presetName of customPresetNames) {
      await page.getByRole('tab', { exact: true, name: presetName }).waitFor();
    }
    assert.equal(await page.getByRole('button', { name: /^More layout presets/ }).count(), 0);
    const presetScroller = page.locator('[data-layout-preset-scroll]');
    await presetScroller.waitFor();
    const presetScrollerMetrics = await presetScroller.evaluate((scroller) => ({
      clientWidth: scroller.clientWidth,
      scrollWidth: scroller.scrollWidth,
    }));
    assert.ok(presetScrollerMetrics.scrollWidth > presetScrollerMetrics.clientWidth);
    const savePresetButton = page.getByRole('button', { exact: true, name: 'Save this layout as a new preset' });
    const [scrollerBounds, savePresetBounds] = await Promise.all([
      presetScroller.boundingBox(),
      savePresetButton.boundingBox(),
    ]);
    assert.ok(scrollerBounds);
    assert.ok(savePresetBounds);
    assert.ok(savePresetBounds.x >= scrollerBounds.x + scrollerBounds.width);
    assert.ok(savePresetBounds.x + savePresetBounds.width <= 1_440);
    const horizontalScroll = await presetScroller.evaluate(async (scroller) => {
      scroller.scrollLeft = scroller.scrollWidth;
      await new Promise((resolveFrame) => {
        requestAnimationFrame(resolveFrame);
      });

      return scroller.scrollLeft;
    });
    assert.ok(horizontalScroll > 0);
    const presetLayoutMetrics = await page.getByRole('banner').evaluate((header) => ({
      clientWidth: header.clientWidth,
      scrollWidth: header.scrollWidth,
    }));
    assert.equal(presetLayoutMetrics.scrollWidth, presetLayoutMetrics.clientWidth);

    await page.getByRole('tab', { name: /^Edit(?:, unsaved changes)?$/ }).click({ button: 'right' });
    const switchLayout = page.getByRole('menuitem', { exact: true, name: 'Switch to this layout' });
    await switchLayout.waitFor();
    assert.equal(await switchLayout.locator('svg.lucide-check').count(), 0);
    assert.equal(await switchLayout.locator('svg.lucide-arrow-right').count(), 1);

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }

    return { id, status: 'passed' };
  } finally {
    await context.close();
  }
};

const runVideoPreviewJourney = async (browser) => {
  const { consoleErrors, context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath);
  const id = 'workbench-video-preview-representative';

  try {
    await waitForWorkbench(page);
    await selectLayoutPreset(page, 'Compose', 'Preview');

    const rightPanel = page.getByRole('complementary', { exact: true, name: 'right widget panel' });
    await rightPanel.locator('button:not([aria-expanded])').filter({ hasText: 'Uncategorized' }).click();
    const gallery = rightPanel.getByRole('list', { exact: true, name: 'Gallery items' });
    const selectVideo = rightPanel.getByRole('button', {
      exact: true,
      name: `Select video ${MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME}, duration 0:01, for preview`,
    });

    try {
      await gallery.waitFor();
    } catch (error) {
      const bodyText = (await page.locator('body').innerText()).slice(0, 4_000);
      throw new Error(
        `${error instanceof Error ? error.message : String(error)}\nPage errors: ${pageErrors.map(String).join('\n')}\nConsole errors: ${consoleErrors.join('\n')}\nBody:\n${bodyText}`
      );
    }
    await selectVideo.waitFor();
    await selectVideo.focus();
    await expectFocused(selectVideo, 'The video gallery item must be keyboard focusable.');
    assert.equal(await selectVideo.locator('xpath=..').locator('svg.lucide-play').getAttribute('aria-hidden'), 'true');
    await selectVideo.press('Enter');

    const video = centerRegion(page).locator(`video[aria-label="Video ${MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME}"]`);

    try {
      await video.waitFor();
    } catch (error) {
      const bodyText = (await page.locator('body').innerText()).slice(0, 4_000);
      throw new Error(
        `${error instanceof Error ? error.message : String(error)}\nPage errors: ${pageErrors.map(String).join('\n')}\nConsole errors: ${consoleErrors.join('\n')}\nBody:\n${bodyText}`
      );
    }
    assert.equal(await video.getAttribute('controls'), '');
    assert.equal(await video.getAttribute('playsinline'), '');
    assert.match((await video.getAttribute('poster')) ?? '', /fixture-video-001\.mp4\/thumbnail$/);
    assert.equal(await video.getAttribute('draggable'), null);

    // Media details live in the floating header chrome's Details popover; audit the surface with it open.
    const details = page
      .locator('[data-hotkey-widget-region="center"][data-hotkey-widget-type-id="preview"]')
      .getByRole('button', { exact: true, name: 'Details' });
    await details.focus();
    await expectFocused(details, 'The preview Details toggle must be keyboard focusable.');
    await details.press('Enter');
    await page.getByText(/Duration 0:01/).waitFor();
    await waitForSettledDocument(page);

    // Generated media has no caption track, so only this video surface disables axe's caption rule.
    await assertNoAxeViolations(page, id, { rules: { 'video-caption': { enabled: false } } });

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }

    return { id, status: 'passed' };
  } finally {
    await context.close();
  }
};

/**
 * Verify retained scroll and canvas state across layout switches; mounted-node identity alone cannot detect state
 * loss.
 */
const runKeepAliveStateJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath);
  const id = 'workbench-keep-alive-state';

  /**
   * Capture both axes of actual scroll containers; clipped text is not scroll state, and filmstrips may overflow
   * only horizontally.
   */
  const SCROLLER_QUERY = `(element) => [...element.querySelectorAll('*')].filter((node) => {
    const style = getComputedStyle(node);
    const scrollsY = (style.overflowY === 'auto' || style.overflowY === 'scroll') && node.scrollHeight > node.clientHeight + 40;
    const scrollsX = (style.overflowX === 'auto' || style.overflowX === 'scroll') && node.scrollWidth > node.clientWidth + 40;
    return scrollsY || scrollsX;
  })`;

  const readScrollOffsets = (panel) =>
    panel.evaluate(
      new Function(
        'element',
        `return (${SCROLLER_QUERY})(element).map((node) => ({ left: Math.round(node.scrollLeft), top: Math.round(node.scrollTop) }));`
      )
    );

  const scrollAll = (panel, offset) =>
    panel.evaluate(
      new Function(
        'element',
        'target',
        `const scrollers = (${SCROLLER_QUERY})(element);
         for (const scroller of scrollers) {
           scroller.scrollTop = target;
           scroller.scrollLeft = target;
         }
         return scrollers.map((scroller) => ({ left: Math.round(scroller.scrollLeft), top: Math.round(scroller.scrollTop) }));`
      ),
      offset
    );

  try {
    await waitForWorkbench(page);
    await selectLayoutPreset(page, 'Compose', 'Preview');

    const rightPanel = page.getByRole('complementary', { exact: true, name: 'right widget panel' });
    await rightPanel.locator('button:not([aria-expanded])').filter({ hasText: 'Uncategorized' }).click();
    const galleryItems = rightPanel.getByRole('list', { exact: true, name: 'Gallery items' });
    await galleryItems.waitFor();

    // Select an unstarred item so the filmstrip exists and has enough entries to overflow.
    await galleryItems
      .locator('[data-gallery-section="regular"]')
      .getByRole('button', { name: /for preview$/ })
      .first()
      .click();

    const leftPanel = page.getByRole('complementary', { exact: true, name: 'left widget panel' });
    const scrolledLeftOffsets = await scrollAll(leftPanel, 400);
    const scrolledOffsets = await scrollAll(rightPanel, 600);
    assert.ok(
      scrolledOffsets.length >= 2 && scrolledOffsets.every(({ left, top }) => top > 0 || left > 0),
      `The representative gallery must have scrollable boards and items; got ${JSON.stringify(scrolledOffsets)}.`
    );

    // Wait for filmstrip overflow: its board-scoped fetch completes independently of the gallery and exercises
    // horizontal scroll retention.
    const centerPanel = centerRegion(page);
    await page.waitForFunction(
      () => {
        const region = [...document.querySelectorAll('[role="region"]')].find(
          (node) => node.getAttribute('aria-label') === 'Center view'
        );

        return region
          ? [...region.querySelectorAll('*')].some((node) => {
              const style = getComputedStyle(node);
              return (
                (style.overflowX === 'auto' || style.overflowX === 'scroll') && node.scrollWidth > node.clientWidth + 40
              );
            })
          : false;
      },
      undefined,
      { timeout: 10_000 }
    );
    const scrolledCenterOffsets = await scrollAll(centerPanel, 300);
    assert.ok(
      scrolledCenterOffsets.some(({ left }) => left > 0),
      `The representative preview filmstrip must scroll horizontally; got ${JSON.stringify(scrolledCenterOffsets)}.`
    );
    await waitForSettledDocument(page);

    await selectLayoutPreset(page, 'Edit', 'Canvas');

    const zoomTrigger = page.getByRole('button', { exact: true, name: 'Zoom level' });
    await zoomTrigger.click();
    await page.getByRole('menuitem', { exact: true, name: '200%' }).click();
    await waitForSettledDocument(page);
    assert.equal((await zoomTrigger.textContent())?.trim(), '200%', 'The zoom control must report the chosen zoom.');

    // Away and back, which is the whole point of keeping the widgets mounted.
    await selectLayoutPreset(page, 'Automate', 'Workflow');
    await waitForSettledDocument(page);
    await selectLayoutPreset(page, 'Edit', 'Canvas');
    await waitForSettledDocument(page);

    assert.equal(
      (await zoomTrigger.textContent())?.trim(),
      '200%',
      'Returning to a layout must keep the canvas viewport, not refit it.'
    );

    await selectLayoutPreset(page, 'Compose', 'Preview');
    await galleryItems.waitFor();
    await waitForSettledDocument(page);

    assert.deepEqual(
      await readScrollOffsets(rightPanel),
      scrolledOffsets,
      'Returning to a layout must keep the gallery scrolled where the user left it.'
    );
    assert.deepEqual(
      await readScrollOffsets(leftPanel),
      scrolledLeftOffsets,
      'Returning to a layout must keep a panel body scrolled where the user left it.'
    );
    assert.deepEqual(
      await readScrollOffsets(centerPanel),
      scrolledCenterOffsets,
      'Returning to a layout must keep the preview filmstrip scrolled where the user left it.'
    );

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }

    return { id, status: 'passed' };
  } finally {
    await context.close();
  }
};

/**
 * Exercise pane switching by pointer and keyboard, tool-dependent Properties, and accessibility of non-default
 * panes.
 */
const LAYERS_PANEL_SCOPE = { include: ['[data-hotkey-widget-type-id="layers"]'] };

const runLayersPanesJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath);

  try {
    await waitForWorkbench(page);
    await selectLayoutPreset(page, 'Edit', 'Canvas');
    const propertiesTab = page.getByRole('tab', { exact: true, name: 'Properties' });
    await propertiesTab.waitFor();

    // Tool switching swaps the Tool section's rows in place.
    const tools = page.getByRole('toolbar', { exact: true, name: 'Tools' });
    await tools.getByRole('button', { exact: true, name: 'Brush' }).click();
    await page
      .getByRole('tabpanel', { exact: true, name: 'Properties' })
      .getByRole('slider', { exact: true, name: 'Brush size' })
      .waitFor();
    await tools.getByRole('button', { exact: true, name: 'View' }).click();
    await page.getByRole('tabpanel', { exact: true, name: 'Properties' }).waitFor();

    // Pointer: activate every non-default pane across the three blocks.
    await page.getByRole('tab', { exact: true, name: 'Overview' }).click();
    await page.getByRole('button', { exact: true, name: 'Pan the canvas view' }).waitFor();
    await page.getByRole('tab', { exact: true, name: 'History' }).click();
    await page.getByRole('tabpanel', { exact: true, name: 'History' }).waitFor();
    await page.getByRole('tab', { exact: true, name: 'Swatches' }).click();
    await page.getByRole('tabpanel', { exact: true, name: 'Swatches' }).waitFor();

    await waitForSettledDocument(page);
    // Scope this audit to Layers so unrelated lazy generation panels cannot change its result.
    await assertNoAxeViolations(page, 'workbench-layers-panes', LAYERS_PANEL_SCOPE);

    // Keyboard: arrows rove within the block's tablist, Enter activates.
    const overviewTab = page.getByRole('tab', { exact: true, name: 'Overview' });
    await overviewTab.focus();
    await overviewTab.press('ArrowLeft');
    const transformTab = page.getByRole('tab', { exact: true, name: 'Transform' });
    await expectFocused(transformTab, 'ArrowLeft should move focus from Overview to Transform.');
    await transformTab.press('Enter');
    await page.getByRole('tabpanel', { exact: true, name: 'Transform' }).waitFor();
    assert.equal(await transformTab.getAttribute('aria-selected'), 'true');

    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, 'workbench-layers-panes:transform', LAYERS_PANEL_SCOPE);

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, 'workbench-layers-panes raised uncaught browser errors.');
    }

    return { id: 'workbench-layers-panes', status: 'passed' };
  } finally {
    await context.close();
  }
};

/**
 * Floating a rail panel that is also the center's only view: the rail keeps a marker, the center says where the
 * view went, and removing or docking the window gives the center its view — and keyboard focus — back.
 */
const runFloatingWindowJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath);
  const id = 'workbench-floating-window';

  try {
    await waitForWorkbench(page);
    await selectLayoutPreset(page, 'Video', 'Preview');

    const rail = page.getByRole('navigation', { exact: true, name: 'Inspect widget visibility' });
    const center = centerRegion(page);
    const floatingWindow = page.locator('[data-hotkey-widget-region="floating"]');
    const marker = rail.getByRole('button', { exact: true, name: 'Preview, floating window' });
    const floatPreviewFromRail = async () => {
      await rail.getByRole('button', { exact: true, name: 'Inspect widgets' }).click();
      await page.getByRole('menuitemcheckbox', { exact: true, name: 'Preview' }).click();
      await page.keyboard.press('Escape');
      await page.getByRole('button', { exact: true, name: 'Float Window' }).click();
      await floatingWindow.waitFor();
      await marker.waitFor();
    };
    const focusedRegion = () =>
      page.evaluate(() => document.activeElement?.closest('[data-focus-region]')?.getAttribute('data-focus-region'));
    // Focus, the accent outline, and the top of the stack all name the same window.
    const waitForActiveWindow = () =>
      page.waitForFunction(() => {
        const active = document.activeElement?.closest('[data-floating-window]');

        return active !== null && active !== undefined && active.getAttribute('data-highlighted') === 'true';
      });
    const waitForFocusedRegion = (region) =>
      page.waitForFunction(
        (expected) =>
          document.activeElement?.closest('[data-focus-region]')?.getAttribute('data-focus-region') === expected,
        region
      );

    await floatPreviewFromRail();
    await waitForActiveWindow();
    await center.getByText('Preview is in a floating window', { exact: true }).waitFor();
    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, `${id}:floating`);

    // The window is operable from the keyboard: the one labelled grip resizes it, and the title bar moves it.
    const width = async () => Math.round((await floatingWindow.boundingBox()).width);
    const left = async () => Math.round((await floatingWindow.boundingBox()).x);
    const windowedWidth = await width();
    await floatingWindow.getByRole('separator', { exact: true, name: 'Resize window' }).focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(
      ([selector, expected]) => Math.round(document.querySelector(selector).getBoundingClientRect().width) === expected,
      ['[data-hotkey-widget-region="floating"]', windowedWidth + 16]
    );
    const windowedLeft = await left();
    await floatingWindow.getByLabel('Move Preview window', { exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(
      ([selector, expected]) => Math.round(document.querySelector(selector).getBoundingClientRect().left) === expected,
      ['[data-hotkey-widget-region="floating"]', windowedLeft + 16]
    );

    // Maximize and collapse are explicit, labelled controls; each state is scanned.
    await floatingWindow.getByRole('button', { exact: true, name: 'Maximize' }).click();
    await floatingWindow.getByRole('button', { exact: true, name: 'Restore' }).waitFor();
    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, `${id}:maximized`);
    await floatingWindow.getByRole('button', { exact: true, name: 'Restore' }).click();
    assert.equal(await width(), windowedWidth + 16, 'Restore must return to the windowed size.');
    await floatingWindow.getByRole('button', { exact: true, name: 'Collapse' }).click();
    await floatingWindow.getByRole('button', { exact: true, name: 'Expand' }).waitFor();
    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, `${id}:collapsed`);
    await floatingWindow.getByRole('button', { exact: true, name: 'Expand' }).click();
    await floatingWindow.getByRole('button', { exact: true, name: 'Collapse' }).waitFor();

    // The marker is a keyboard stop, and it shows the window rather than docking it.
    await marker.focus();
    await page.keyboard.press('Enter');
    assert.equal(await floatingWindow.count(), 1, 'Activating the marker must keep the window floating.');
    await waitForActiveWindow();

    await marker.click({ button: 'right' });
    assert.deepEqual(await page.getByRole('menuitem').allInnerTexts(), ['Dock to right panel', 'Remove Preview']);
    await page.getByRole('menuitem', { exact: true, name: 'Remove Preview' }).click();
    await centerViewTrigger(page, 'Preview').waitFor();
    await floatingWindow.waitFor({ state: 'detached' });
    assert.equal(await marker.count(), 0, 'Removing the window must free its rail slot.');
    await waitForFocusedRegion('center');

    await floatPreviewFromRail();
    await waitForActiveWindow();
    await center.getByRole('button', { exact: true, name: 'Dock to right panel' }).click();
    await centerViewTrigger(page, 'Preview').waitFor();
    await floatingWindow.waitFor({ state: 'detached' });
    await rail.getByRole('button', { exact: true, name: 'Preview' }).waitFor();
    await waitForFocusedRegion('center');
    assert.equal(await focusedRegion(), 'center');

    // Docking brought Preview to the front of the right panel. Float it again: docking from the window's own
    // control returns focus to that panel.
    await page.getByRole('button', { exact: true, name: 'Float Window' }).click();
    await floatingWindow.waitFor();
    await waitForActiveWindow();
    await floatingWindow.getByRole('button', { exact: true, name: 'Dock to right panel' }).click();
    await floatingWindow.waitFor({ state: 'detached' });
    await waitForFocusedRegion('right');

    // The rail's own dock path: the marker's menu, reached from the keyboard, with focus following the panel.
    await page.getByRole('button', { exact: true, name: 'Float Window' }).click();
    await floatingWindow.waitFor();
    await waitForActiveWindow();
    await marker.focus();
    await page.keyboard.press('Shift+F10');
    await page.getByRole('menuitem', { exact: true, name: 'Dock to right panel' }).click();
    await floatingWindow.waitFor({ state: 'detached' });
    await waitForFocusedRegion('right');
    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, `${id}:docked`);

    // Removing a window from its marker's menu takes the menu and the marker with it. Focus goes to the panel the
    // window's rail is showing, or to the center when that rail shows none: collapsed, or emptied by the float.
    const floatImageMapFrom = async (widgetRail, menuName) => {
      await widgetRail.getByRole('button', { exact: true, name: menuName }).click();
      await page.getByRole('menuitemcheckbox', { exact: true, name: 'Image Map' }).click();
      await page.keyboard.press('Escape');
      await page.getByRole('button', { exact: true, name: 'Float Window' }).click();
      await floatingWindow.waitFor();
      await waitForActiveWindow();
    };
    // From the keyboard, where losing focus costs the most: Remove is the menu's last item.
    const removeImageMapFrom = async (widgetRail, focusedRegionAfter) => {
      await widgetRail.getByRole('button', { exact: true, name: 'Image Map, floating window' }).focus();
      await page.keyboard.press('Shift+F10');
      await page.waitForFunction(() => Boolean(document.activeElement?.closest('[role="menu"]')));
      await page.keyboard.press('End');
      await page.keyboard.press('Enter');
      await floatingWindow.waitFor({ state: 'detached' });
      await waitForFocusedRegion(focusedRegionAfter);
    };

    await floatImageMapFrom(rail, 'Inspect widgets');
    await removeImageMapFrom(rail, 'right');

    await floatImageMapFrom(rail, 'Inspect widgets');
    await rail.getByRole('button', { pressed: true }).click();
    await page.getByRole('complementary', { exact: true, name: 'right widget panel' }).waitFor({ state: 'detached' });
    await removeImageMapFrom(rail, 'center');

    const leftRail = page.getByRole('navigation', { exact: true, name: 'Create widget visibility' });
    for (const name of ['Video', 'Upscale']) {
      await leftRail.getByRole('button', { exact: true, name }).click({ button: 'right' });
      await page.getByRole('menuitem', { exact: true, name: `Remove ${name}` }).click();
    }
    await floatImageMapFrom(leftRail, 'Create widgets');
    await page.getByRole('complementary', { exact: true, name: 'left widget panel' }).waitFor({ state: 'detached' });
    await removeImageMapFrom(leftRail, 'center');

    // The enable menu stays open across a toggle and keeps focus; removing a window from it moves none.
    await floatImageMapFrom(leftRail, 'Create widgets');
    await leftRail.getByRole('button', { exact: true, name: 'Create widgets' }).click();
    await page.getByRole('menuitemcheckbox', { exact: true, name: 'Image Map' }).click();
    await floatingWindow.waitFor({ state: 'detached' });
    // A focus move would have landed within a frame of the removal.
    await page.evaluate(
      () =>
        new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        })
    );
    assert.equal(
      await page.evaluate(() => Boolean(document.activeElement?.closest('[role="menu"]'))),
      true,
      'Removing a window from the enable menu must leave focus in that menu.'
    );
    await page.keyboard.press('Escape');

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }

    return { id, status: 'passed' };
  } finally {
    await context.close();
  }
};

const SETTINGS_DIALOG_SCOPE = { include: ['[data-scope="dialog"][data-part="content"]'] };

const runSettingsJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, representativeProjectPath);
  const id = 'workbench-settings';

  try {
    await waitForWorkbench(page);
    const gear = page.getByRole('button', { exact: true, name: 'Gallery settings' });
    await gear.click();
    // Wait for lazy settings controls; their arrival moves the footer's click target.
    await page.getByRole('slider', { exact: true, name: 'Image size' }).waitFor();
    await waitForSettledDocument(page);
    await page.getByRole('button', { exact: true, name: 'All Gallery settings…' }).click();
    const dialog = page.getByRole('dialog', { name: /^Settings:/ });
    await dialog.waitFor();
    const navigation = dialog.getByRole('navigation', { exact: true, name: 'Settings' });
    const sectionNames = await navigation.getByRole('button').allTextContents();
    assert.equal(
      sectionNames.length,
      15,
      'All application, project, widget, and system sections must be discoverable.'
    );
    assert(sectionNames.includes('Intermediates'), 'Intermediates settings must be discoverable.');
    for (const name of sectionNames) {
      await navigation.getByRole('button', { exact: true, name }).click();
      await page.getByRole('dialog', { exact: true, name: `Settings: ${name}` }).waitFor();
      await page.waitForLoadState('networkidle');
      await waitForSettledDocument(page);
      // Certify the settings surface; representative surface scans retain page-wide workbench coverage.
      await assertNoAxeViolations(page, `${id}:${name}`, SETTINGS_DIALOG_SCOPE);
    }

    const search = dialog.getByRole('textbox', { exact: true, name: 'Search settings…' });
    await search.focus();
    await search.pressSequentially('numeric attention');
    const numericAttention = dialog.getByRole('checkbox', { exact: true, name: 'Prefer numeric attention style' });
    await numericAttention.waitFor();
    await expectFocused(search, 'Filtering settings should keep keyboard focus in search.');
    await numericAttention.focus();
    await numericAttention.press('Space');
    assert.equal(await numericAttention.isChecked(), true);
    await waitForSettledDocument(page);
    await assertNoAxeViolations(page, `${id}:search`, SETTINGS_DIALOG_SCOPE);
    await dialog.getByRole('button', { exact: true, name: 'Show in section' }).click();
    await expectFocused(numericAttention, 'Revealing a setting should focus its control.');
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'hidden' });
    await expectFocused(gear, 'Closing settings should return focus to its widget gear.');

    await gear.press('Enter');
    const popover = page.locator('[data-scope="popover"][data-part="content"][data-state="open"]');
    await popover.waitFor();
    await waitForSettledDocument(page);
    // Quick settings are nonmodal; scan their popup independently of the workbench behind it.
    await assertNoAxeViolations(page, `${id}:quick`, { include: ['[data-scope="popover"][data-part="content"]'] });
    await page.keyboard.press('Escape');
    await popover.waitFor({ state: 'hidden' });
    await expectFocused(gear, 'Closing quick settings should restore focus to its widget gear.');

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }
    return { id, status: 'passed' };
  } finally {
    await context.close();
  }
};

const mockBackend = await startMockBackend(backendPort, { profile: 'representative' });
const preview = spawnPreview({
  cwd: root,
  env: { ...process.env, INVOKEAI_DEV_BACKEND: backendOrigin },
  port,
  stdio: ['ignore', 'pipe', 'pipe'],
});
let previewError = '';
let browser = null;

preview.stderr.on('data', (chunk) => {
  previewError += String(chunk);
});

try {
  await waitForPreview();
  browser = await chromium.launch({ headless: true });
  const reports = [];

  for (const surface of surfaces.filter(({ id }) => !requestedJourney || id === requestedJourney)) {
    reports.push(await runAxeSurface(browser, surface));
  }

  if (!requestedJourney || requestedJourney === 'keyboard-critical-journey') {
    reports.push(await runKeyboardJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-shortcuts') {
    reports.push(await runShortcutGuideJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-topbar-responsive') {
    reports.push(await runResponsiveTopbarJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-topbar-menus') {
    reports.push(await runTopbarMenuJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-video-preview-representative') {
    reports.push(await runVideoPreviewJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-keep-alive-state') {
    reports.push(await runKeepAliveStateJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-layers-panes') {
    reports.push(await runLayersPanesJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-floating-window') {
    reports.push(await runFloatingWindowJourney(browser));
  }
  if (!requestedJourney || requestedJourney === 'workbench-settings') {
    reports.push(await runSettingsJourney(browser));
  }
  if (reports.length === 0) {
    throw new Error(`Unknown accessibility journey ${JSON.stringify(requestedJourney)}.`);
  }
  process.stdout.write(`${JSON.stringify({ profile: 'representative', reports }, null, 2)}\n`);
} catch (error) {
  throw new Error(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}${previewError ? `\n${previewError}` : ''}`
  );
} finally {
  await browser?.close();

  if (preview.pid) {
    try {
      killPreview(preview.pid, 'SIGTERM');
    } catch {
      // Preview may already have exited after a startup failure.
    }
  }

  await mockBackend.close();
}
