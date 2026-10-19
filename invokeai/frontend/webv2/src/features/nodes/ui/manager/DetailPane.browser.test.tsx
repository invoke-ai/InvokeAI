import { ChakraProvider } from '@chakra-ui/react';
import { setCustomNodesSnapshotForTests } from '@features/nodes/data/nodesStore';
import { openNodePackDetail } from '@features/nodes/ui/nodesUiStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { DetailPane } from './DetailPane';

const api = vi.hoisted(() => ({
  getPackWorkflowCount: vi.fn(() => Promise.resolve(0)),
  uninstallCustomNodePack: vi.fn(() => Promise.resolve({ message: 'Uninstalled', name: 'pack-a' })),
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

const PACK = { name: 'pack-a', nodeCount: 0, nodeTypes: [], path: '/custom_nodes/pack-a' };

describe('nodes DetailPane', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    accountLifecycle.activate('nodes-detail-pane-test', ':user:nodes-detail-pane-test');
    setCustomNodesSnapshotForTests({ nodePacks: [PACK], status: 'loaded' });
    openNodePackDetail(PACK.name);
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <DetailPane />
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('animates the uninstall confirmation out after the uninstalled pack leaves the pane', async () => {
    const uninstall = [...document.querySelectorAll('button')].find(
      (button) => button.textContent === 'nodes.uninstall'
    );
    await act(() => userEvent.click(uninstall!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="alertdialog"]')!;
    const confirm = [...dialog.querySelectorAll('button')].find(
      (button) => button.textContent === 'nodes.uninstallPack'
    );

    const frames = closingFrames(
      await recordDialogExit(dialog, async () => {
        await act(() => userEvent.click(confirm!));
        await expect.poll(() => host.textContent).toContain('nodes.selectPack');
      })
    );

    expect(api.uninstallCustomNodePack).toHaveBeenCalledWith(PACK.name, expect.anything());
    expect(frames).not.toHaveLength(0);
    // It still names the pack it uninstalled while it animates out.
    for (const frame of frames) {
      expect(frame.text).toContain('nodes.uninstallTitlepack-a');
    }
  });
});
