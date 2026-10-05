import { ChakraProvider } from '@chakra-ui/react';
import { setCustomNodesSnapshotForTests } from '@features/nodes/data/nodesStore';
import { closeNodePackDetail, updateNodesUi } from '@features/nodes/ui/nodesUiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { NodeManagerView } from './NodeManagerView';

const api = vi.hoisted(() => ({
  getPackWorkflowCount: vi.fn(() => Promise.resolve(0)),
  uninstallCustomNodePack: vi.fn(() => Promise.resolve({ message: 'Uninstalled', name: 'pack-b' })),
}));

vi.mock('@features/nodes/data/api', async (importOriginal) => ({ ...(await importOriginal<object>()), ...api }));
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  ensureInvocationTemplatesLoaded: vi.fn(),
  useInvocationTemplatesSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, status: 'loaded', templates: {} }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, options?: { name?: string }) => `${key}${options?.name ?? ''}` }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const PACKS = ['pack-a', 'pack-b', 'pack-c'].map((name) => ({
  name,
  nodeCount: 0,
  nodeTypes: [],
  path: `/custom_nodes/${name}`,
}));
/** The Launchpad's page area at 720×450 CSS px (1440×900 at 200%): beside the icon rail, below the top bar. */
const ZOOMED_PAGE = { height: 402, width: 675 };

const back = () => page.getByRole('button', { name: 'nodes.backToList' });
const library = () => page.getByRole('heading', { name: 'nodes.nodePacks' });
const detail = () => page.getByRole('tablist');
const row = (name: string) => document.querySelector<HTMLElement>(`[data-list-row="${name}"] [data-list-primary]`)!;

describe('node manager in a single pane', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    applyThemeToRoot(DEFAULT_THEME_ID);
    accountLifecycle.activate('node-manager-view-test', ':user:node-manager-view-test');
    setCustomNodesSnapshotForTests({ nodePacks: PACKS, status: 'loaded' });
    // Setting a tab opens the detail; start from the library.
    updateNodesUi({ activePackName: null, activeTab: 'add' });
    closeNodePackDetail();
    host = document.createElement('div');
    host.style.cssText = `display:flex;height:${String(ZOOMED_PAGE.height)}px;width:${String(ZOOMED_PAGE.width)}px;`;
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <NodeManagerView />
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('opens a pack, reaches Uninstall, and returns to the pack row', async () => {
    await expect.element(library()).toBeVisible();
    await expect.element(detail()).not.toBeInTheDocument();
    expect(await auditAccessibility(host)).toEqual([]);

    await userEvent.click(row('pack-b'));

    await expect.element(back()).toHaveFocus();
    const uninstall = page.getByRole('button', { name: 'nodes.uninstall', exact: true });
    await expect.element(uninstall).toBeVisible();
    const bounds = uninstall.element().getBoundingClientRect();
    const hostBounds = host.getBoundingClientRect();
    expect(bounds.right).toBeLessThanOrEqual(hostBounds.right);
    expect(bounds.bottom).toBeLessThanOrEqual(hostBounds.bottom);
    await uninstall.click();
    await expect.element(page.getByRole('alertdialog')).toBeVisible();
    await userEvent.keyboard('{Escape}');
    // Escape closed the dialog; it did not also leave the detail.
    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    await expect.element(detail()).toBeVisible();
    expect(await auditAccessibility(host)).toEqual([]);

    await back().click();

    await expect.element(library()).toBeVisible();
    expect(document.activeElement).toBe(row('pack-b'));
  });

  it('returns to the library with focus on a row after uninstalling the open pack', async () => {
    await userEvent.click(row('pack-b'));

    await page.getByRole('button', { name: 'nodes.uninstall', exact: true }).click();
    await page.getByRole('alertdialog').getByRole('button', { name: 'nodes.uninstallPack' }).click();

    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    await expect.element(library()).toBeVisible();
    await vi.waitFor(() => {
      const focused = document.activeElement as HTMLElement;
      expect(focused.matches('[data-list-primary]')).toBe(true);
      expect(focused.checkVisibility({ visibilityProperty: true })).toBe(true);
    });
    expect(api.uninstallCustomNodePack).toHaveBeenCalledWith('pack-b', expect.anything());
  });
});

describe('node manager starting pane', () => {
  let host: HTMLDivElement;
  let root: Root;

  const addTab = () => page.getByRole('tab', { name: 'nodes.addNodes' });

  const mount = async (packs: typeof PACKS) => {
    applyThemeToRoot(DEFAULT_THEME_ID);
    // An account change resets the manager, so the starting pane is still undecided.
    accountLifecycle.activate('node-manager-start-test', ':user:node-manager-start-test');
    setCustomNodesSnapshotForTests({ nodePacks: packs, status: 'loaded' });
    host = document.createElement('div');
    host.style.cssText = `display:flex;height:${String(ZOOMED_PAGE.height)}px;width:${String(ZOOMED_PAGE.width)}px;`;
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <NodeManagerView />
        </ChakraProvider>
      )
    );
  };

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('opens a non-empty library on the list', async () => {
    await mount(PACKS);

    await expect.element(library()).toBeVisible();
    await expect.element(addTab()).not.toBeInTheDocument();
  });

  it('opens an empty library on Add Nodes, and an arriving pack leaves the pane, tab and focus alone', async () => {
    await mount([]);

    await expect.element(addTab()).toHaveAttribute('aria-selected', 'true');
    await expect.element(library()).not.toBeInTheDocument();
    expect(document.activeElement).toBe(document.body);
    const field = await vi.waitFor(() => {
      const input = document.querySelector<HTMLInputElement>('[role="tabpanel"] input');
      expect(input).not.toBeNull();
      return input!;
    });
    await act(() => userEvent.click(field));

    await act(() => setCustomNodesSnapshotForTests({ nodePacks: [PACKS[0]!], status: 'loaded' }));

    await expect.element(addTab()).toHaveAttribute('aria-selected', 'true');
    await expect.element(library()).not.toBeInTheDocument();
    expect(document.activeElement).toBe(field);
  });

  it('stays on the list, now empty, when the last pack goes', async () => {
    await mount([PACKS[0]!]);
    await expect.element(library()).toBeVisible();

    await act(() => setCustomNodesSnapshotForTests({ nodePacks: [], status: 'loaded' }));

    await expect.element(library()).toBeVisible();
    await expect.element(addTab()).not.toBeInTheDocument();
  });
});
