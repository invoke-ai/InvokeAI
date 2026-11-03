import type * as HttpModule from '@platform/transport/http';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@platform/ui/theme/applyTheme';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mocks = vi.hoisted(() => ({ apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof HttpModule>()),
  apiFetchJson: (...args: unknown[]) => mocks.apiFetchJson(...args),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

const preview = (operation: string) => ({
  operation,
  fingerprint: 'a'.repeat(64),
  examined_count: 42,
  affected_count: 7,
  skipped_count: 1,
  error_count: 0,
  errors: [],
  archive_path: '/var/lib/invokeai/gallery_archive/2026-10-07/run-01',
});

const settleDialogMotion = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 250);
  });

let host: HTMLDivElement | undefined;
let root: Root | undefined;
let queryClient: QueryClient | undefined;

const renderSettings = async (width: number, height: number): Promise<void> => {
  await page.viewport(width, height);
  host = document.createElement('div');
  host.style.cssText = 'box-sizing:border-box;min-height:700px;padding:24px;width:100%;';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { GalleryMaintenance } = await import('./GalleryMaintenance');
  await act(() => {
    root!.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <QueryClientProvider client={queryClient!}>
            <Box bg="bg.subtle" p="4">
              <GalleryMaintenance />
            </Box>
          </QueryClientProvider>
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

beforeEach(() => {
  accountLifecycle.activate('gallery-maintenance-visual', ':user:gallery-maintenance-visual');
  applyThemeToRoot('light');
  mocks.apiFetchJson.mockReset().mockImplementation((path: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as { operation: string };
    return Promise.resolve(preview(body.operation));
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient?.clear();
  host?.remove();
  accountLifecycle.invalidate();
  applyThemeToRoot('classic');
  root = undefined;
  queryClient = undefined;
  host = undefined;
});

describe('Gallery maintenance visual evidence', () => {
  it('shows localized described rows and each destructive confirmation in the light theme', async () => {
    await renderSettings(1280, 900);
    await expect.element(page.getByRole('heading', { name: 'Gallery maintenance' })).toBeVisible();
    await expect
      .element(
        page.getByText(
          'Remove gallery entries whose original image files are missing. Remaining thumbnails are moved to the archive.',
          { exact: true }
        )
      )
      .toBeVisible();
    await page.screenshot({ path: '../../../artifacts/gallery-maintenance/settings-light-wide.png' });

    for (const [label, title, confirm] of [
      ['Remove records for missing images', 'Remove records for missing images?', 'Remove missing image records'],
      ['Archive untracked files', 'Archive untracked image files?', 'Archive untracked files'],
      ['Regenerate missing thumbnails', 'Regenerate missing thumbnails?', 'Regenerate thumbnails'],
    ]) {
      await page.getByRole('button', { name: label, exact: true }).click();
      const dialog = page.getByRole('alertdialog');
      await expect.poll(() => dialog.element().textContent).toContain(title);
      await expect.poll(() => dialog.element().textContent).toContain('7 affected');
      await expect
        .poll(() => dialog.element().textContent)
        .toContain('/var/lib/invokeai/gallery_archive/2026-10-07/run-01');
      const confirmButton = page.getByRole('button', { name: confirm, exact: true });
      await expect.element(confirmButton).toBeVisible();
      await settleDialogMotion();
      await page.screenshot({
        path: `../../../artifacts/gallery-maintenance/dialog-${confirm.toLowerCase().replaceAll(' ', '-')}-light.png`,
      });
      await page.getByRole('button', { name: 'Cancel', exact: true }).click();
      await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    }
  });

  it('shows a narrow dark-theme layout and a destructive confirmation', async () => {
    applyThemeToRoot('ultradark');
    await renderSettings(520, 800);
    await page.screenshot({ path: '../../../artifacts/gallery-maintenance/settings-dark-narrow.png' });
    await page.getByRole('button', { name: 'Remove records for missing images', exact: true }).click();
    await settleDialogMotion();
    await expect
      .poll(() => page.getByRole('alertdialog').element().textContent)
      .toContain('This maintenance scans images across all accounts');
    await page.screenshot({ path: '../../../artifacts/gallery-maintenance/dialog-dark-narrow.png' });
  });

  it('captures preview loading and recovery error states', async () => {
    let resolvePreview!: (value: ReturnType<typeof preview>) => void;
    mocks.apiFetchJson
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolvePreview = resolve;
        })
      )
      .mockRejectedValueOnce(new Error('The preview could not read one image folder.'));
    await renderSettings(900, 800);
    await page.getByRole('button', { name: 'Remove records for missing images', exact: true }).click();
    await expect.poll(() => page.getByRole('status').element().textContent).toContain('Scanning for affected images');
    await settleDialogMotion();
    await page.screenshot({ path: '../../../artifacts/gallery-maintenance/preview-loading.png' });
    await act(async () => {
      resolvePreview(preview('remove_missing'));
      await Promise.resolve();
    });
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();

    await page.getByRole('button', { name: 'Archive untracked files', exact: true }).click();
    await expect
      .poll(() => page.getByRole('alert').element().textContent)
      .toContain('The preview could not read one image folder');
    await settleDialogMotion();
    await page.screenshot({ path: '../../../artifacts/gallery-maintenance/preview-error-retry.png' });
  });
});
