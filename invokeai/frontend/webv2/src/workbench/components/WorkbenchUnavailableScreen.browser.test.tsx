import type { SyncedWorkbenchPersistence } from '@workbench/projects/syncedPersistence';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  downloadText: vi.fn(),
  getRecoverableDraftDocument: vi.fn(() => Promise.resolve<string | null>('{"id":"project-1"}')),
  listRecoverableDrafts: vi.fn(() =>
    Promise.resolve({
      items: [
        {
          documentByteSize: 18,
          editorSessionId: 'editor-1',
          projectId: 'project-1',
          state: 'dirty' as const,
          updatedAt: 1,
        },
      ],
      kind: 'available' as const,
      nextCursor: null,
    })
  ),
}));
vi.mock('@platform/browser/downloadBlob', () => ({ downloadText: harness.downloadText }));
vi.mock('@workbench/queue-integration/QueueRecoveryNotice', () => ({
  QueueRecoveryNotice: () => <div data-testid="queue-recovery-notice">queue recovery</div>,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { WorkbenchUnavailableScreen } from './WorkbenchUnavailableScreen';

const persistence = {
  getRecoverableDraftDocument: harness.getRecoverableDraftDocument,
  listRecoverableDrafts: harness.listRecoverableDrafts,
} as unknown as SyncedWorkbenchPersistence;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  harness.downloadText.mockClear();
  harness.getRecoverableDraftDocument.mockReset();
  harness.getRecoverableDraftDocument.mockResolvedValue('{"id":"project-1"}');
  harness.listRecoverableDrafts.mockReset();
  harness.listRecoverableDrafts.mockResolvedValue({
    items: [
      {
        documentByteSize: 18,
        editorSessionId: 'editor-1',
        projectId: 'project-1',
        state: 'dirty',
        updatedAt: 1,
      },
    ],
    kind: 'available',
    nextCursor: null,
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WorkbenchUnavailableScreen', () => {
  it('offers retry and exports durable drafts while the editor remains blocked', async () => {
    const onRetry = vi.fn();
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <WorkbenchUnavailableScreen message="offline" persistence={persistence} onRetry={onRetry} />
        </ChakraProvider>
      );
      await new Promise((resolve) => {
        window.setTimeout(resolve, 0);
      });
    });
    expect(document.body.textContent).toContain('project-1');
    const buttons = Array.from(document.querySelectorAll('button'));

    await act(() => userEvent.click(buttons.find((button) => button.textContent?.includes('retry'))!));
    await act(() => userEvent.click(buttons.find((button) => button.textContent?.includes('exportDraft'))!));

    expect(onRetry).toHaveBeenCalledOnce();
    expect(harness.downloadText).toHaveBeenCalledWith('{"id":"project-1"}', 'project-1.json', 'application/json');
    expect(document.querySelector('[data-testid="queue-recovery-notice"]')).not.toBeNull();
  });

  it('does not misreport inaccessible recovery storage as an empty draft list', async () => {
    harness.listRecoverableDrafts.mockResolvedValueOnce({ kind: 'unavailable' } as never);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <WorkbenchUnavailableScreen message="offline" persistence={persistence} onRetry={vi.fn()} />
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 0);
      });
    });

    expect(document.body.textContent).toContain('shell.backendUnavailable.draftsUnavailable');
    expect(document.body.textContent).not.toContain('shell.backendUnavailable.noDrafts');
  });

  it('reports an export that can no longer read its draft', async () => {
    harness.getRecoverableDraftDocument.mockResolvedValueOnce(null);
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <WorkbenchUnavailableScreen message="offline" persistence={persistence} onRetry={vi.fn()} />
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 0);
      });
    });
    const exportButton = Array.from(document.querySelectorAll('button')).find((button) =>
      button.textContent?.includes('exportDraft')
    );

    await act(() => userEvent.click(exportButton!));
    await vi.waitFor(() => expect(document.body.textContent).toContain('shell.backendUnavailable.draftUnavailable'));

    expect(harness.downloadText).not.toHaveBeenCalled();
  });
});
