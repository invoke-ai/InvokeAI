import type { InstallNodePackResponse } from '@features/nodes/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import {
  addCustomNodeInstallLogEntry,
  getCustomNodeInstallLogForTests,
  updateCustomNodeInstallLogEntry,
} from '@features/nodes/data/installLogStore';
import { setCustomNodesSnapshotForTests } from '@features/nodes/data/nodesStore';
import { updateNodesUi, useNodesUiSelector } from '@features/nodes/ui/nodesUiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { NodeInstallQueue } from './NodeInstallQueue';

const api = vi.hoisted(() => ({
  installCustomNodePack: vi.fn(),
  listCustomNodePacks: vi.fn(),
}));

vi.mock('@features/nodes/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...api,
}));

vi.mock('@features/nodes/ui/useNodesNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn(), warning: vi.fn() }),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const FAILED_SOURCE = 'https://github.com/example/broken-pack.git';
const PACK = { name: 'pack-a', nodeCount: 2, nodeTypes: [], path: '/custom_nodes/pack-a' };

let root: Root | undefined;
let host: HTMLDivElement | undefined;

/** Shows the manager state the queue's openers write to. */
const ManagerProbe = () => {
  const activeTab = useNodesUiSelector((snapshot) => snapshot.activeTab);
  const activePackName = useNodesUiSelector((snapshot) => snapshot.activePackName);

  return <output data-testid="probe">{`${activeTab}:${activePackName ?? ''}`}</output>;
};

const render = async () => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;flex-direction:column;height:600px;width:1000px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ManagerProbe />
          <NodeInstallQueue />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const rows = () => [...(host?.querySelectorAll<HTMLTableRowElement>('tbody tr[data-install-status]') ?? [])];
const rowByStatus = (status: string) => rows().find((row) => row.dataset.installStatus === status);
const findButton = (name: string, within: ParentNode | undefined = host) =>
  [...(within?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find(
    (candidate) => candidate.getAttribute('aria-label') === name || candidate.textContent?.trim() === name
  );
const click = (name: string, within: ParentNode | undefined = host) =>
  act(() => {
    const button = findButton(name, within);

    if (!button) {
      throw new Error(`No button named ${name}`);
    }

    button.click();
  });
const header = () => host?.querySelector('[aria-expanded]')?.parentElement;

/** Oldest first, so the log lists them newest first: installing, failed, installed, uninstalled. */
const seedLog = () => {
  addCustomNodeInstallLogEntry({
    message: "Successfully uninstalled node pack 'old-pack'.",
    name: 'old-pack',
    status: 'uninstalled',
  });
  const done = addCustomNodeInstallLogEntry({
    name: 'https://github.com/example/pack-a.git',
    source: 'https://github.com/example/pack-a.git',
    status: 'installing',
  });
  updateCustomNodeInstallLogEntry(done.id, {
    message: "Successfully installed node pack 'pack-a'.",
    name: 'pack-a',
    status: 'completed',
  });
  addCustomNodeInstallLogEntry({
    message: 'Git clone failed: repository not found',
    name: 'broken-pack',
    source: FAILED_SOURCE,
    status: 'error',
  });
};

beforeEach(async () => {
  await page.viewport(1100, 700);
  applyThemeToRoot(DEFAULT_THEME_ID);
  // Resets the account-owned install log and manager UI state.
  accountLifecycle.invalidate();
  accountLifecycle.activate('node-install-queue-test');
  setCustomNodesSnapshotForTests({ nodePacks: [PACK], status: 'loaded' });
  for (const mock of Object.values(api)) {
    mock.mockReset();
  }
  api.listCustomNodePacks.mockResolvedValue({ customNodesPath: '/custom_nodes', nodePacks: [PACK] });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;
});

describe('NodeInstallQueue', () => {
  it('renders each status with its badge, caption, detail, and only the actions it supports', async () => {
    seedLog();
    addCustomNodeInstallLogEntry({
      name: 'https://github.com/example/new-pack.git',
      source: 'https://github.com/example/new-pack.git',
      status: 'installing',
    });
    updateNodesUi({ queueExpanded: true });
    await render();

    expect(rows().map((row) => row.dataset.installStatus)).toEqual(['installing', 'error', 'completed', 'uninstalled']);
    expect(header()?.textContent).toContain('1 installing');
    expect(header()?.textContent).toContain('1 needs attention');
    expect(header()?.textContent).toContain('1 installed');
    expect(header()?.textContent).toContain('1 uninstalled');

    const installing = rowByStatus('installing')!;
    expect(installing.textContent).toContain('Cloning repository');
    expect(installing.textContent).toContain('Installing');
    expect(installing.querySelector('[role="progressbar"]')).not.toBeNull();
    expect(installing.querySelectorAll('button')).toHaveLength(0);

    const failed = rowByStatus('error')!;
    expect(failed.dataset.tone).toBe('error');
    expect(failed.textContent).toContain('Failed');
    expect(failed.textContent).toContain(FAILED_SOURCE);
    expect(failed.textContent).toContain('Git clone failed: repository not found');
    // Installs run one at a time, so a retry waits for the running one and says why.
    const retry = findButton('Retry install', failed)!;
    expect(retry.getAttribute('aria-disabled')).toBe('true');
    await expect
      .element(page.elementLocator(retry))
      .toHaveAccessibleDescription('Another node pack is installing. Retry when it finishes.');
    await act(() => retry.click());
    expect(api.installCustomNodePack).not.toHaveBeenCalled();
    expect(findButton('Remove from list', failed)).toBeDefined();

    const installed = rowByStatus('completed')!;
    expect(installed.textContent).toContain('Installed');
    expect(installed.textContent).toContain('Done');
    expect(installed.textContent).toContain("Successfully installed node pack 'pack-a'.");
    expect(findButton('Open node pack', installed)).toBeDefined();
    expect(findButton('Retry install', installed)).toBeUndefined();

    const uninstalled = rowByStatus('uninstalled')!;
    expect(uninstalled.textContent).toContain('Uninstalled');
    expect(uninstalled.textContent).toContain('Restart to apply');
    expect(findButton('Open node pack', uninstalled)).toBeUndefined();
    expect(findButton('Remove from list', uninstalled)).toBeDefined();

    expect(await auditAccessibility(host!)).toEqual([]);
  });

  it('retries a failed install from its source in place of the failed row', async () => {
    seedLog();
    let resolveInstall: (response: InstallNodePackResponse) => void = () => undefined;
    api.installCustomNodePack.mockImplementation(
      () =>
        new Promise<InstallNodePackResponse>((resolve) => {
          resolveInstall = resolve;
        })
    );
    updateNodesUi({ queueExpanded: true });
    await render();

    await click('Retry install', rowByStatus('error'));

    expect(api.installCustomNodePack).toHaveBeenCalledWith(FAILED_SOURCE, expect.anything());
    expect(rows().map((row) => row.dataset.installStatus)).toEqual(['installing', 'completed', 'uninstalled']);
    expect(rowByStatus('installing')?.textContent).toContain(FAILED_SOURCE);

    await act(() => {
      resolveInstall({
        dependency_file: null,
        message: "Successfully installed node pack 'broken-pack'.",
        name: 'broken-pack',
        requires_dependencies: false,
        success: true,
        workflows_imported: 0,
      });
    });

    await vi.waitFor(() => {
      expect(rows().map((row) => row.dataset.installStatus)).toEqual(['completed', 'completed', 'uninstalled']);
    });
    expect(rows()[0]?.textContent).toContain('broken-pack');
    expect(rows()[0]?.textContent).toContain(FAILED_SOURCE);
  });

  it('features the running install while collapsed, and clearing finished keeps it', async () => {
    seedLog();
    addCustomNodeInstallLogEntry({
      name: 'https://github.com/example/new-pack.git',
      source: 'https://github.com/example/new-pack.git',
      status: 'installing',
    });
    await render();

    expect(header()?.textContent).toContain('https://github.com/example/new-pack.git');
    expect(header()?.textContent).toContain('1 needs attention');
    expect(header()?.textContent).toContain('1 installed');
    expect(header()?.querySelector('[role="progressbar"]')).not.toBeNull();

    await click('Clear finished');

    expect(getCustomNodeInstallLogForTests().map((entry) => entry.status)).toEqual(['installing']);
    expect(findButton('Clear finished')).toBeUndefined();
  });

  it('removes one settled row from the list', async () => {
    seedLog();
    updateNodesUi({ queueExpanded: true });
    await render();

    await click('Remove from list', rowByStatus('uninstalled'));

    expect(rows().map((row) => row.dataset.installStatus)).toEqual(['error', 'completed']);
  });

  it('shows the empty state with nothing to clear', async () => {
    updateNodesUi({ queueExpanded: true });
    await render();

    expect(host?.textContent).toContain('No installs yet');
    expect(host?.textContent).toContain('Node pack installs and uninstalls appear here.');
    expect(findButton('Clear finished')?.disabled).toBe(true);
  });

  it('expands to fill the pane, restores, and opening a pack restores it too', async () => {
    seedLog();
    updateNodesUi({ queueExpanded: true, queueMaximized: false });
    await render();

    await click('Expand');
    expect(findButton('Restore')).toBeDefined();
    await click('Restore');
    expect(findButton('Expand')).toBeDefined();

    // Row navigation must uncover the pane it targets.
    await click('Expand');
    await click('Open node pack', rowByStatus('completed'));
    expect(findButton('Expand')).toBeDefined();
    expect(host?.querySelector('[data-testid="probe"]')?.textContent).toBe('details:pack-a');
  });
});
