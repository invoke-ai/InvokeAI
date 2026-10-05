import type { Project } from '@workbench/projectContracts';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { act, createRef, useImperativeHandle, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useFindGalleryItem } from './useFindGalleryItem';

const mocks = vi.hoisted(() => ({
  opened: [] as { options: unknown; widgetId: string }[],
  project: null as unknown as Project,
  reveal: vi.fn(() => Promise.resolve()),
}));

vi.mock('@workbench/useOpenWorkbenchWidget', () => ({
  useOpenWorkbenchWidget: () => (widgetId: string, options: unknown) => {
    mocks.opened.push({ options, widgetId });
    return { ok: true, region: 'right' };
  },
}));
vi.mock('@workbench/WorkbenchContext', () => {
  const queries = {
    getSnapshot: () => ({ activeProject: mocks.project }),
    isActiveProject: (id: string) => id === mocks.project.id,
  };
  const commands = { notifications: { reportError: vi.fn() } };

  return { useWorkbenchCommands: () => commands, useWorkbenchQueries: () => queries };
});
vi.mock('@workbench/image-actions/revealGalleryItem', () => ({ revealGalleryItem: mocks.reveal }));

type Find = ReturnType<typeof useFindGalleryItem>;
const findRef = createRef<Find>();
const Probe = ({ ref }: { ref: Ref<Find> }) => {
  const find = useFindGalleryItem();

  useImperativeHandle(ref, () => find, [find]);

  return null;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** The default project with a Gallery shown in the center as well as the right panel. */
const projectWithCenterGallery = (): Project => {
  const project = structuredClone(createInitialWorkbenchState().projects[0]!);
  const galleryId = Object.values(project.widgetInstances).find((instance) => instance.typeId === 'gallery')!.id;

  project.widgetRegions.center = {
    ...project.widgetRegions.center,
    instanceIds: [...project.widgetRegions.center.instanceIds, galleryId],
  };

  return project;
};

beforeEach(async () => {
  mocks.opened = [];
  mocks.reveal.mockClear();
  mocks.project = projectWithCenterGallery();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <Probe ref={findRef} />
      </QueryClientProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('useFindGalleryItem', () => {
  it('brings Preview to the center and the Gallery forward wherever it is, by default', async () => {
    await act(() => findRef.current!({ kind: 'image', name: 'found.png' }));

    expect(mocks.opened.map(({ widgetId }) => widgetId)).toEqual(['preview', 'gallery']);
    await vi.waitFor(() => expect(mocks.reveal).toHaveBeenCalledOnce());
  });

  it('leaves the center alone when asked, raising the Gallery only in a panel', async () => {
    await act(() => findRef.current!({ kind: 'image', name: 'found.png' }, { revealPreview: false }));

    expect(mocks.opened).toHaveLength(1);
    expect(mocks.opened[0]!.widgetId).toBe('gallery');
    expect((mocks.opened[0]!.options as { preferredRegions: string[] }).preferredRegions).not.toContain('center');
    await vi.waitFor(() =>
      expect(mocks.reveal).toHaveBeenCalledWith(
        expect.anything(),
        { kind: 'image', name: 'found.png' },
        expect.objectContaining({ projectId: mocks.project.id })
      )
    );
  });
});
