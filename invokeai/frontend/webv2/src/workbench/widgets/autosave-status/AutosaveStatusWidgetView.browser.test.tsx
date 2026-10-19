import type { AutosaveState } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { reportProjectSync } from '@workbench/projects/syncStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({
  autosave: { status: 'idle' } as AutosaveState,
  projects: [
    { id: 'portraits', name: 'Portraits' },
    { id: 'landscapes', name: 'Landscapes' },
  ],
}));

// The frame barrel reaches the whole workbench context; the chip itself is the platform control.
vi.mock('@workbench/widget-frame', async () => ({
  StatusWidgetChip: (await import('@platform/ui/StatusWidgetChip')).StatusWidgetChip,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ autosave: harness.autosave, projects: harness.projects }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string; reason?: string; status?: string }) =>
      options?.name ? `${options.name}: ${options.reason}` : options?.status ? `${key}(${options.status})` : key,
  }),
}));

import { AutosaveStatusWidgetView } from './AutosaveStatusWidgetView';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (presentation: 'compact' | 'tooltip'): Promise<HTMLDivElement> => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <AutosaveStatusWidgetView {...({ presentation, region: 'bottom' } as WidgetViewProps)} />
      </ChakraProvider>
    )
  );
  return host!;
};

const reportSync = (projects: Parameters<typeof reportProjectSync>[0]['projects']): void =>
  reportProjectSync({ hasPendingChanges: true, localDraftStatus: 'ok', projects, recoverableDrafts: [] });

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  reportProjectSync({ hasPendingChanges: false, localDraftStatus: 'ok', projects: {}, recoverableDrafts: [] });
});

describe('AutosaveStatusWidgetView', () => {
  it('names each open project awaiting a decision and why, instead of reporting an error', async () => {
    harness.autosave = { status: 'attention' };
    reportSync({
      // A project no longer open is not listed: the user cannot switch to it to resolve it.
      closed: {
        conflict: { detectedAt: '2026-10-14T09:00:00.000Z', kind: 'deleted' },
        isPendingPush: true,
        revision: 2,
      },
      landscapes: { isPendingPush: false, revision: 4 },
      portraits: {
        conflict: { detectedAt: '2026-10-14T09:00:00.000Z', kind: 'revision', serverRevision: 7 },
        isPendingPush: true,
        revision: 6,
      },
    });

    expect((await render('compact')).textContent).toBe(
      'widgets.autosaveStatus.chipLabel(widgets.autosaveStatus.attention)'
    );

    const items = [...(await render('tooltip')).querySelectorAll('li')].map((item) => item.textContent);
    expect(items).toEqual(['Portraits: widgets.autosaveStatus.reasonRevision']);
    expect(host?.textContent).not.toContain('widgets.autosaveStatus.error');
  });

  it('reports a schema refusal as needing a newer client', async () => {
    harness.autosave = { status: 'attention' };
    reportSync({
      landscapes: {
        isPendingPush: true,
        revision: 4,
        schemaRefusal: { kind: 'canvas', maxCanvasSchemaVersion: 1, minimumCanvasSchemaVersion: 2 },
      },
    });

    const items = [...(await render('tooltip')).querySelectorAll('li')].map((item) => item.textContent);
    expect(items).toEqual(['Landscapes: widgets.autosaveStatus.reasonSchema']);
  });

  it('shows a retrying save as unsaved changes without surfacing the internal reason', async () => {
    harness.autosave = { error: 'Autosave is pending and will retry.', status: 'pending' };

    expect((await render('compact')).textContent).toBe(
      'widgets.autosaveStatus.chipLabel(widgets.autosaveStatus.pending)'
    );
    const tooltip = (await render('tooltip')).textContent;
    expect(tooltip).toContain('widgets.autosaveStatus.retrying');
    expect(tooltip).not.toContain('Autosave is pending and will retry.');
  });

  it('names a conflicted open project while another project is still retrying', async () => {
    harness.autosave = { error: 'Autosave is pending and will retry.', status: 'pending' };
    reportSync({
      landscapes: { isPendingPush: true, revision: 4 },
      portraits: {
        conflict: { detectedAt: '2026-10-14T09:00:00.000Z', kind: 'deleted' },
        isPendingPush: true,
        revision: 6,
      },
    });

    const tooltip = await render('tooltip');
    const items = [...tooltip.querySelectorAll('li')].map((item) => item.textContent);
    expect(items).toEqual(['Portraits: widgets.autosaveStatus.reasonDeleted']);
    expect(tooltip.textContent).toContain('widgets.autosaveStatus.retrying');
  });

  it('keeps the failure message for a save that failed', async () => {
    harness.autosave = { error: 'Backend rejected the revision', status: 'error' };

    expect((await render('compact')).textContent).toBe(
      'widgets.autosaveStatus.chipLabel(widgets.autosaveStatus.error)'
    );
    expect((await render('tooltip')).textContent).toContain('Backend rejected the revision');
  });
});
