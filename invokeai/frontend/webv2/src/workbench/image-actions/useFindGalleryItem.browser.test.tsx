import type { Project } from '@workbench/projectContracts';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { act, createRef, useImperativeHandle, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useFindGalleryItem } from './useFindGalleryItem';

const mocks = vi.hoisted(() => ({
  opened: [] as unknown[],
  project: null as unknown as Project,
  reveal: vi.fn(() => Promise.resolve()),
}));

vi.mock('@workbench/workspace/useWorkspaceOpen', () => ({
  useWorkspaceOpen: () => (intents: unknown, options: unknown) => {
    mocks.opened.push([intents, options]);
    return { changedWorkspace: false, ok: true, region: 'right', workspaceId: 'compose' };
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

beforeEach(async () => {
  mocks.opened = [];
  mocks.reveal.mockClear();
  mocks.project = createInitialWorkbenchState().projects[0]!;
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
  it('shows Preview on the stage and then the Gallery in one gesture, focusing the Gallery, by default', async () => {
    await act(() => findRef.current!({ kind: 'image', name: 'found.png' }));

    expect(mocks.opened).toEqual([
      [[{ surface: 'stage', typeId: 'preview' }, { typeId: 'gallery' }], { focus: 'gallery' }],
    ]);
    await vi.waitFor(() => expect(mocks.reveal).toHaveBeenCalledOnce());
  });

  it('leaves the center alone when asked, showing the Gallery only as a panel', async () => {
    await act(() => findRef.current!({ kind: 'image', name: 'found.png' }, { revealPreview: false }));

    expect(mocks.opened).toEqual([[{ surface: 'panel', typeId: 'gallery' }, { focus: 'gallery' }]]);
    await vi.waitFor(() =>
      expect(mocks.reveal).toHaveBeenCalledWith(
        expect.anything(),
        { kind: 'image', name: 'found.png' },
        expect.objectContaining({ projectId: mocks.project.id })
      )
    );
  });
});
