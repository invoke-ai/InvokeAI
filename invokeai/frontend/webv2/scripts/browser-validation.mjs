/** Strict nominal-journey diagnostics; expected failures are scenario-local predicates. */
export const observeBrowserDiagnostics = (page, { expected = () => false, onError = () => {} } = {}) => {
  const errors = [];
  const media = [];
  const responseStatuses = new WeakMap();
  const pendingConsoleDetails = new Set();
  const socket404 = (url, status) => {
    const parsed = new URL(url);
    return parsed.pathname === '/ws/socket.io/' && parsed.searchParams.get('transport') === 'polling' && status === 404;
  };
  const record = (event) => {
    if (!expected(event)) {
      errors.push(event);
      onError(event);
    }
  };
  page.on('pageerror', (error) => record({ kind: 'pageerror', message: error.stack ?? error.message }));
  page.on('console', (message) => {
    if (message.type() !== 'error') {
      return;
    }
    const url = message.location().url;
    if (url && socket404(url, 404) && /(?:404|Not Found)/.test(message.text())) {
      return;
    }
    const event = { kind: 'console', url, message: message.text() };
    record(event);
    // Firefox labels Error objects only as "Error"; retain the actionable message and stack before navigation.
    const details = Promise.all(
      (message.args?.() ?? []).map((argument) =>
        argument
          .evaluate((value) =>
            value && typeof value.message === 'string'
              ? { name: value.name, message: value.message, stack: value.stack }
              : null
          )
          .catch(() => null)
      )
    ).then((arguments_) => {
      event.arguments = arguments_.filter(Boolean);
    });
    pendingConsoleDetails.add(details);
    void details.finally(() => pendingConsoleDetails.delete(details));
  });
  page.on('requestfailed', (request) => {
    // Queue polling is canceled by its owner on route exit; an HTTP failure still fails below.
    if (
      new URL(request.url()).pathname === '/api/v1/queue/default/status' &&
      /(?:ERR_ABORTED|NS_BINDING_ABORTED|cancelled)/i.test(request.failure()?.errorText ?? '')
    ) {
      return;
    }
    record({
      kind: 'requestfailed',
      url: request.url(),
      method: request.method?.(),
      message: request.failure()?.errorText,
      responseStatus: responseStatuses.get(request),
    });
  });
  page.on('response', (response) => {
    const request = response.request?.();
    if (request) {
      responseStatuses.set(request, response.status());
    }
    const url = response.url();
    if (/\/api\/v1\/(images|videos)\/i\/.*\/(full|thumbnail)/.test(url)) {
      media.push({ url, status: response.status() });
    }
    if (response.status() >= 400 && !socket404(url, response.status())) {
      record({ kind: 'response', url, status: response.status() });
    }
  });
  return {
    errors,
    media,
    async assertHealthy() {
      await Promise.all(pendingConsoleDetails);
      if (errors.length) {
        throw new Error(`Unexpected browser diagnostics:\n${JSON.stringify(errors, null, 2)}`);
      }
    },
  };
};

/** Mount marks are separate: this proves visible decoded images and raster pixels, without product test flags. */
export const waitForRenderedMedia = async (page, { canvas = false, requireImage = false } = {}) => {
  await page.waitForFunction(
    ({ canvasRequired, imageRequired }) => {
      const visible = (element) => element.checkVisibility() && element.getBoundingClientRect().width > 0;
      const images = [...document.images].filter(visible);
      if (imageRequired && images.length === 0) {
        return false;
      }
      if (images.some((image) => !image.complete)) {
        return false;
      }
      const broken = images.find((image) => image.naturalWidth === 0);
      if (broken) {
        throw new Error(`Visible media failed to decode: ${broken.currentSrc || broken.src}`);
      }
      if (!canvasRequired) {
        return true;
      }
      const hasPixels = (selector, fixtureColours) =>
        [...document.querySelectorAll(selector)].filter(visible).some((surface) => {
          if (!surface.width || !surface.height) {
            return false;
          }
          const pixels = surface.getContext('2d')?.getImageData(0, 0, surface.width, surface.height).data;
          if (!pixels) {
            return false;
          }
          for (let i = 0; i < pixels.length; i += 4) {
            if (
              fixtureColours
                ? (pixels[i] === 64 && pixels[i + 1] === 160) || (pixels[i] === 192 && pixels[i + 1] === 80)
                : pixels[i + 3] > 0
            ) {
              return true;
            }
          }
          return false;
        });
      return (
        hasPixels('[data-hotkey-widget-type-id="canvas"] canvas', true) &&
        hasPixels('[data-hotkey-widget-type-id="layers"] canvas', false)
      );
    },
    { canvasRequired: canvas, imageRequired: requireImage },
    { timeout: 20_000 }
  );
  return page.evaluate(async (canvasRequired) => {
    const images = [...document.images].filter((image) => image.checkVisibility());
    await Promise.all(images.map((image) => image.decode()));
    return {
      images: images.map((image) => ({
        url: image.currentSrc,
        width: image.naturalWidth,
        height: image.naturalHeight,
      })),
      canvasPixelsVerified: canvasRequired,
    };
  }, canvas);
};

export const assertRetirement = (samples, bounds) => {
  for (const [key, maximum] of Object.entries(bounds)) {
    if (samples.some((sample) => sample[key] > maximum)) {
      throw new Error(`Retained ${key} exceeded ${maximum}: ${samples.map((sample) => sample[key]).join(', ')}`);
    }
    if (samples.length >= 4 && samples.every((sample, index) => index === 0 || sample[key] > samples[index - 1][key])) {
      throw new Error(`Monotonic retained ${key} growth: ${samples.map((sample) => sample[key]).join(', ')}`);
    }
  }
};

export const distribution = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    count: sorted.length,
    min: sorted[0] ?? 0,
    median: sorted[Math.floor(sorted.length / 2)] ?? 0,
    p95: sorted[Math.max(0, Math.ceil(sorted.length * 0.95) - 1)] ?? 0,
    max: sorted.at(-1) ?? 0,
  };
};

/** Warm one quick-settings popover, then six complete open/interact/close cycles. Portals must retire each time. */
export const runSustainedWorkload = async (page) => {
  const gear = page.getByRole('button', { exact: true, name: 'Gallery settings' });
  const popup = page.locator('[data-scope="popover"][data-part="content"][data-state="open"]');
  const requests = [];
  const onRequest = (request) => requests.push(request.url());
  page.on('request', onRequest);
  const samples = [];
  const ownedPopoverIds = [];
  const longTaskStart = await page.evaluate(() => window.__architectureLongTasks?.length ?? 0);
  try {
    for (let cycle = 0; cycle < 7; cycle += 1) {
      const start = performance.now();
      await gear.click();
      await page.getByRole('slider', { exact: true, name: 'Image size' }).waitFor();
      const popupId = await page
        .getByRole('slider', { exact: true, name: 'Image size' })
        .evaluate((element) => element.closest('[data-scope="popover"][data-part="content"]').id);
      if (!popupId) {
        throw new Error('Quick settings popover has no identity.');
      }
      if (!ownedPopoverIds.includes(popupId)) {
        ownedPopoverIds.push(popupId);
      }
      await page.getByRole('slider', { exact: true, name: 'Image size' }).focus();
      await page.keyboard.press(cycle % 2 ? 'ArrowLeft' : 'ArrowRight');
      await page.keyboard.press('Escape');
      await popup.waitFor({ state: 'hidden' });
      await page.waitForFunction((ids) => ids.every((id) => !document.getElementById(id)), ownedPopoverIds);
      const retained = await page.evaluate(
        (ids) => ({
          domNodes: document.querySelectorAll('*').length,
          popovers: ids.filter((id) => document.getElementById(id)).length,
          canvases: document.querySelectorAll('canvas').length,
        }),
        ownedPopoverIds
      );
      if (cycle > 0) {
        samples.push({ ...retained, cycle, latencyMs: performance.now() - start, requests: requests.length });
      }
    }
    assertRetirement(samples, { popovers: 0 });
    return {
      workload: 'gallery-quick-settings',
      warmups: 1,
      cycles: 6,
      samples,
      interactionMs: distribution(samples.map((sample) => sample.latencyMs)),
      longTasksMs: distribution(
        await page.evaluate((start) => (window.__architectureLongTasks ?? []).slice(start), longTaskStart)
      ),
      timingEnforced: false,
    };
  } catch (error) {
    error.sustained = { workload: 'gallery-quick-settings', samples };
    throw error;
  } finally {
    page.off('request', onRequest);
  }
};
