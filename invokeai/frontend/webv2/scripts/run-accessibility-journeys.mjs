import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright';

import { assertNoAxeViolations } from './accessibility/axe.mjs';
import { MOCK_BACKEND_REPRESENTATIVE_VIDEO_NAME } from './mock-backend-fixtures.mjs';
import { startMockBackend } from './mock-backend.mjs';

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
    await new Promise((resolveFrame) => {
      requestAnimationFrame(() => requestAnimationFrame(resolveFrame));
    });
  });
};

const openRepresentativePage = async (browser, path, viewport = { height: 1_000, width: 1_440 }) => {
  const resetResponse = await fetch(`${backendOrigin}/__reset?profile=representative`, { method: 'POST' });

  if (!resetResponse.ok) {
    throw new Error(`Could not reset representative backend: ${resetResponse.status}.`);
  }

  const context = await browser.newContext({
    colorScheme: 'dark',
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

/**
 * `/projects` is the library itself, which is where the grid lives. The
 * heading is matched at level 2 because the shell also renders a visually
 * hidden level-1 heading naming the active section, which on this page is the
 * same word.
 */
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

/**
 * Presets are a segmented radio group, and they name an arrangement rather than
 * a widget — so the center view a preset lands on is passed explicitly instead
 * of being assumed to share its name.
 */
const selectLayoutPreset = async (page, preset, centerView) => {
  // Anchored at both ends, because custom presets share the strip: a loose
  // `^Edit` also matches a user's "Edit copy". The optional suffix is the drift
  // marker the tab folds into its own accessible name.
  const name = new RegExp(`^${preset}(, unsaved changes)?$`);
  const selected = page.getByRole('tab', { name, selected: true });

  if ((await selected.count()) === 0) {
    await page.getByRole('tab', { name }).click();
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
    },
  },
  {
    id: 'workbench-gallery-representative',
    path: representativeProjectPath,
    ready: async (page) => {
      // Gallery is a widget, not an arrangement, so it is reached through the
      // center area's own picker rather than by loading a preset named after it.
      await waitForWorkbench(page);
      await selectLayoutPreset(page, 'Compose', 'Preview');
      await selectCenterView(page, 'Preview', 'Gallery');
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

/**
 * Roving-tabindex and focus-restore moves land on a later task than the key
 * press that causes them, so sampling `document.activeElement` once races the
 * behaviour under test. Poll to a deadline instead: the assertion is unchanged
 * (focus must reach this element) but it no longer depends on scheduling.
 */
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

const runKeyboardJourney = async (browser) => {
  const { context, page, pageErrors } = await openRepresentativePage(browser, '/#/');

  try {
    await waitForHome(page);

    const projectsTab = page.getByRole('tab', { exact: true, name: 'Projects' });
    const modelsTab = page.getByRole('tab', { exact: true, name: 'Models' });

    // The rail is grouped, but it is still one tablist: arrowing off the last
    // Workspace tab has to land on the first Manage tab, skipping the headings.
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

    // The center view selector is a menu button: it opens on the keyboard,
    // exposes each view as a radio item, and restores focus to itself on close.
    const previewTrigger = centerViewTrigger(page, 'Preview');

    await previewTrigger.focus();
    await previewTrigger.press('Enter');

    const previewItem = page.getByRole('menuitemradio', { exact: true, name: 'Preview' });
    const canvasItem = page.getByRole('menuitemradio', { exact: true, name: 'Canvas' });
    const centerViewMenu = page.getByRole('menu');

    await canvasItem.waitFor();
    assert.equal(await previewItem.getAttribute('aria-checked'), 'true');
    await expectFocused(centerViewMenu, 'Opening the center view menu should focus its composite.');
    assert.equal(await centerViewMenu.getAttribute('aria-activedescendant'), await previewItem.getAttribute('id'));
    await centerViewMenu.press('ArrowDown');
    assert.equal(await centerViewMenu.getAttribute('aria-activedescendant'), await canvasItem.getAttribute('id'));
    await centerViewMenu.press('Enter');
    const canvasTrigger = centerViewTrigger(page, 'Canvas');
    await canvasTrigger.waitFor();
    await expectFocused(canvasTrigger, 'Selecting a center view should restore focus to the view selector.');

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

    const leftWidgetRail = page.getByRole('navigation', { exact: true, name: 'Left widget visibility' });
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
    const settingsItem = page.getByRole('menuitem', { exact: true, name: 'Settings' });
    const documentationItem = page.getByRole('menuitem', { exact: true, name: 'Documentation' });
    const discordItem = page.getByRole('menuitem', { exact: true, name: 'Discord' });
    await commandPaletteItem.waitFor();
    await settingsItem.waitFor();
    await documentationItem.waitFor();
    await discordItem.waitFor();

    const footerMetrics = await appMenu.evaluate((menu) => {
      const menuBounds = menu.getBoundingClientRect();
      const footerItems = [...menu.querySelectorAll('[role="menuitem"]')].slice(-4);

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
    assert.equal(await appMenu.getAttribute('aria-activedescendant'), await discordItem.getAttribute('id'));
    for (const item of [documentationItem, settingsItem, commandPaletteItem]) {
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
    // Compose puts the Gallery in the right rail and the Preview in the centre,
    // so the video is picked on the right and plays in the middle.
    await waitForWorkbench(page);
    await selectLayoutPreset(page, 'Compose', 'Preview');

    const rightPanel = page.getByRole('complementary', { exact: true, name: 'right widget panel' });
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
    await page.getByText(/Duration 0:01/).waitFor();
    await waitForSettledDocument(page);

    // Generated media has no caption track, so only this video-specific surface
    // disables axe's caption rule. Every other release surface keeps it enabled.
    await assertNoAxeViolations(page, id, { rules: { 'video-caption': { enabled: false } } });

    if (pageErrors.length > 0) {
      throw new AggregateError(pageErrors, `${id} raised uncaught browser errors.`);
    }

    return { id, status: 'passed' };
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

  for (const surface of surfaces.filter(({ id }) => !requestedJourney || id === requestedJourney)) {
    reports.push(await runAxeSurface(browser, surface));
  }

  if (!requestedJourney || requestedJourney === 'keyboard-critical-journey') {
    reports.push(await runKeyboardJourney(browser));
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
      process.kill(-preview.pid, 'SIGTERM');
    } catch {
      // Preview may already have exited after a startup failure.
    }
  }

  await mockBackend.close();
}
