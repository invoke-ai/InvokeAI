import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { DEFAULT_LOGGING_CONFIG, type LoggingConfig } from '@platform/logging/contracts';
import { clearLogs, configureLogging, createLogger, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { AppToaster } from '@platform/ui/toaster';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { DiagnosticsPanel } from './DiagnosticsPanel';

const download = vi.hoisted(() => ({ downloadText: vi.fn() }));

vi.mock('@platform/browser/downloadBlob', () => download);

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | undefined;
let host: HTMLDivElement | undefined;
const originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');

const settle = () =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 20);
    });
  });

const configure = (overrides: Partial<LoggingConfig> = {}) =>
  configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'trace', ...overrides });

const render = async (projectId: string | null, width = '720px') => {
  if (!host) {
    host = document.createElement('div');
    host.style.display = 'flex';
    host.style.flexDirection = 'column';
    host.style.height = '600px';
    document.body.append(host);
    root = createRoot(host);
  }
  host.style.width = width;
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <AppToaster />
          <DiagnosticsPanel projectId={projectId} />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  await settle();
};

const stubClipboard = (writeText: (text: string) => Promise<void>) => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
};

const seed = () => {
  createLogger({ area: 'boot', namespace: 'app' }).info('Application starting');
  createLogger({ area: 'autosave', namespace: 'persistence', projectId: 'project-a' }).error({
    error: new Error('revision rejected'),
    message: 'Autosave failed',
    name: 'persistence.autosave-failed',
  });
  createLogger({ area: 'history', namespace: 'queue', projectId: 'project-a' }).debug('Queue item pending');
  createLogger({ area: 'history', namespace: 'queue', projectId: 'project-b' }).warn('Other project warning');
};

const entryRows = () => [...(host?.querySelectorAll<HTMLElement>('[data-log-entry-id]') ?? [])];
const combobox = (name: string) => page.getByRole('combobox', { name, exact: true });
const choose = async (name: string, option: string) => {
  await act(() => combobox(name).click());
  await expect.element(page.getByRole('option', { name: option, exact: true })).toBeVisible();
  await act(() => page.getByRole('option', { name: option, exact: true }).click());
  await settle();
};

beforeEach(async () => {
  await page.viewport(1200, 800);
  applyThemeToRoot(DEFAULT_THEME_ID);
  resetLogging();
  configure();
  download.downloadText.mockReset();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;
  if (originalClipboard) {
    Object.defineProperty(navigator, 'clipboard', originalClipboard);
  }
  resetLogging();
});

describe('DiagnosticsPanel', () => {
  it('shows project plus application events by default and switches scope', async () => {
    seed();
    await render('project-a');

    expect(entryRows().map((row) => row.dataset.logLevel)).toEqual(['debug', 'error', 'info']);
    expect(host?.textContent).toContain('Recording Trace+ in 15 of 15 namespaces');
    expect(host?.textContent).toContain('2/500 problems');

    await choose('Scope', 'Current project');
    expect(entryRows().map((row) => row.dataset.logLevel)).toEqual(['debug', 'error']);

    await choose('Scope', 'All account events');
    expect(entryRows()).toHaveLength(4);
    expect(host?.textContent).toContain('project-b');

    await expect(auditAccessibility(host!)).resolves.toEqual([]);
  });

  it('applies display-only severity, namespace and text filters without changing recording', async () => {
    seed();
    await render('project-a');

    await choose('Minimum severity', 'Warn and above');
    expect(entryRows().map((row) => row.dataset.logLevel)).toEqual(['error']);
    expect(getLogSnapshot().entries).toHaveLength(4);

    await choose('Minimum severity', 'All severities');
    await choose('Namespace', 'queue');
    expect(entryRows()).toHaveLength(1);
    expect(host?.textContent).toContain('Queue item pending');

    await act(() => page.getByRole('textbox', { name: 'Search events' }).fill('revision'));
    await settle();
    expect(entryRows()).toHaveLength(0);
    expect(host?.textContent).toContain('No events match the current filters.');

    await act(() => page.getByRole('button', { name: 'Clear filters' }).click());
    await settle();
    expect(entryRows()).toHaveLength(3);
  });

  it('pages newest-first at 100 entries and expands raw details with the keyboard', async () => {
    const logger = createLogger({ area: 'history', namespace: 'queue', projectId: 'project-a' });

    for (let index = 0; index < 150; index += 1) {
      logger.warn(`Warning ${index}`);
    }
    await render('project-a');

    expect(entryRows()).toHaveLength(100);
    expect(host?.textContent).toContain('Warning 149');
    expect(host?.textContent).not.toContain('Warning 49');
    expect(host?.textContent).toContain('Page 1 of 2');

    await act(() => page.getByRole('button', { name: 'Next page' }).click());
    await settle();
    expect(entryRows()).toHaveLength(50);
    expect(host?.textContent).toContain('Page 2 of 2');
    expect(host?.textContent).toContain('Warning 0');

    const details = entryRows()[0]?.querySelector<HTMLButtonElement>('button[aria-expanded]');

    details?.focus();
    await act(() => userEvent.keyboard('{Enter}'));
    await settle();
    expect(entryRows()[0]?.querySelector('[data-log-entry-details]')?.textContent).toContain('"name": "history"');
    expect(details?.getAttribute('aria-expanded')).toBe('true');
  });

  it('copies and downloads the filtered view as a versioned envelope and reports failures', async () => {
    seed();
    const writeText = vi.fn().mockResolvedValue(undefined);

    stubClipboard(writeText);
    await render('project-a');
    await choose('Minimum severity', 'Warn and above');

    await act(() => page.getByRole('button', { name: 'Copy JSON (1)' }).click());
    await settle();
    const envelope = JSON.parse(writeText.mock.calls[0]?.[0] as string) as {
      capture: LoggingConfig;
      entries: { message: string }[];
      format: string;
      scope: unknown;
      version: number;
    };

    expect(envelope).toMatchObject({
      capture: { level: 'trace' },
      format: 'invokeai-webv2-logs',
      scope: { kind: 'project-and-application', projectId: 'project-a' },
      version: 1,
    });
    expect(envelope.entries.map((entry) => entry.message)).toEqual(['Autosave failed']);
    await expect.element(page.getByText('Copied 1 event as JSON')).toBeVisible();

    await act(() => page.getByRole('button', { name: 'Download JSON (1)' }).click());
    await settle();
    expect(download.downloadText).toHaveBeenCalledWith(
      expect.stringContaining('"format": "invokeai-webv2-logs"'),
      expect.stringMatching(/^invoke-logs-.*\.json$/),
      'application/json'
    );
    await expect.element(page.getByText('Downloaded 1 event as JSON')).toBeVisible();

    stubClipboard(vi.fn().mockRejectedValue(new Error('denied')));
    download.downloadText.mockImplementation(() => {
      throw new Error('blocked');
    });
    await act(() => page.getByRole('button', { name: 'Copy JSON (1)' }).click());
    await act(() => page.getByRole('button', { name: 'Download JSON (1)' }).click());
    await settle();
    await expect.element(page.getByText('Could not copy events to the clipboard.')).toBeVisible();
    await expect.element(page.getByText('Could not download the events.')).toBeVisible();
  });

  it('clears only the selected scope independently of display filters', async () => {
    seed();
    await render('project-a');
    await act(() => page.getByRole('textbox', { name: 'Search events' }).fill('nothing-matches'));
    await settle();
    expect(entryRows()).toHaveLength(0);

    await act(() => page.getByRole('button', { name: 'Clear: Current project + application' }).click());
    await settle();

    expect(getLogSnapshot().entries.map((entry) => entry.message)).toEqual(['Other project warning']);
    // An empty scope reads as empty even while a text filter is set; the filter no longer matters.
    expect(host?.textContent).toContain('No events recorded in this scope yet.');
    expect(host?.textContent).not.toContain('Clear filters');
  });

  it('distinguishes disabled recording from an empty history and follows project switches', async () => {
    configure({ enabled: false });
    await render('project-a');

    expect(host?.textContent).toContain('Recording off');
    expect(host?.textContent).toContain('Turn on “Record diagnostic logs”');
    await expect.element(page.getByRole('button', { name: 'Open developer settings' })).toBeVisible();

    configure();
    seed();
    await settle();
    expect(entryRows()).toHaveLength(3);

    await render('project-b');
    expect(entryRows().map((row) => row.dataset.logLevel)).toEqual(['warn', 'info']);

    clearLogs();
    await settle();
    expect(host?.textContent).toContain('No events recorded in this scope yet.');
  });

  it('stays usable in a narrow dock and in the dark theme', async () => {
    seed();
    createLogger({ area: 'upload', namespace: 'gallery', projectId: 'project-a' }).warn(
      `Uploaded ${'a-very-long-file-name-'.repeat(8)}.png to the board`
    );
    applyThemeToRoot('ultradark');
    try {
      await render('project-a', '340px');

      expect(host!.scrollWidth).toBeLessThanOrEqual(host!.clientWidth);
      await expect.element(combobox('Scope')).toBeVisible();
      await expect.element(page.getByRole('button', { name: 'Copy JSON (4)' })).toBeVisible();
      await expect(auditAccessibility(host!)).resolves.toEqual([]);
    } finally {
      applyThemeToRoot(DEFAULT_THEME_ID);
    }
  });
});
