/* eslint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop -- the probe wires fakes directly */
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { workflowFitViewRequestStore } from '@features/workflow/ui/editor/flowInstanceStore';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useOpenLibraryWorkflow } from './useOpenLibraryWorkflow';

const library = vi.hoisted(() => ({
  beforeAnswer: () => {},
  getLibraryWorkflowRecordCached: vi.fn(),
}));

const identity = vi.hoisted(() => ({
  session: {
    multiuserEnabled: false,
    phase: 'ready',
    user: null,
  } as {
    multiuserEnabled: boolean;
    phase: 'ready';
    user: { is_admin?: boolean; user_id: string } | null;
  },
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

vi.mock('@features/workflow/queries', () => ({
  getLibraryWorkflowRecordCached: (workflowId: string) => {
    library.beforeAnswer();
    return library.getLibraryWorkflowRecordCached(workflowId);
  },
  touchLibraryWorkflowOpenedAt: () => Promise.resolve(),
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ITEM = { name: 'Portrait', workflow_id: 'lib-portrait' };
const RECORD = {
  created_at: '2026-10-20T00:00:00Z',
  is_public: false,
  name: 'Portrait',
  revision: 4,
  updated_at: '2026-10-20T00:00:00Z',
  user_id: 'user-a',
  workflow: {
    author: '',
    contact: '',
    description: '',
    edges: [],
    exposedFields: [],
    form: { elements: {}, rootElementId: '' },
    meta: { category: 'user', version: '3.0.0' },
    name: 'Portrait',
    nodes: [
      {
        data: {
          id: 'value',
          inputs: { value: { label: '', name: 'value', value: 1 } },
          isIntermediate: false,
          isOpen: true,
          label: '',
          nodePack: 'invokeai',
          notes: '',
          type: 'integer',
          useCache: true,
          version: '1.0.1',
        },
        id: 'value',
        position: { x: 400, y: 300 },
        type: 'invocation',
      },
    ],
    notes: '',
    tags: '',
    version: '3.0.0',
  },
  workflow_id: ITEM.workflow_id,
};

describe('useOpenLibraryWorkflow replace', () => {
  let host: HTMLDivElement;
  let root: Root;
  let projectId: string;
  let commands: { addWorkflow: ReturnType<typeof vi.fn>; replaceWorkflow: ReturnType<typeof vi.fn> };
  let notifications: { error: ReturnType<typeof vi.fn>; info: ReturnType<typeof vi.fn> };
  const onOpened = vi.fn();

  const Probe = () => {
    const { open, replace } = useOpenLibraryWorkflow(onOpened);

    return (
      <>
        <button type="button" onClick={() => void open(ITEM, 'first-copy')}>
          open
        </button>
        <button type="button" onClick={() => void replace(ITEM, 'copy-1')}>
          replace
        </button>
      </>
    );
  };

  const runReplace = async () => {
    await act(() =>
      root.render(
        <WorkflowUiProvider
          adapter={
            {
              commands,
              notifications,
              project: { getSnapshot: () => ({ id: projectId, workflows: [] }), subscribe: () => () => {} },
            } as unknown as WorkflowUiAdapter
          }
        >
          <Probe />
        </WorkflowUiProvider>
      )
    );
    await act(async () => {
      [...host.querySelectorAll<HTMLButtonElement>('button')]
        .find((button) => button.textContent === 'replace')!
        .click();
      await vi.waitFor(() => expect(library.getLibraryWorkflowRecordCached).toHaveBeenCalled());
      // The load waits two frames for the busy overlay to paint before applying.
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      });
    });
  };

  const runOpen = async () => {
    await act(() =>
      root.render(
        <WorkflowUiProvider
          adapter={
            {
              commands,
              notifications,
              project: { getSnapshot: () => ({ id: projectId, workflows: [] }), subscribe: () => () => {} },
            } as unknown as WorkflowUiAdapter
          }
        >
          <Probe />
        </WorkflowUiProvider>
      )
    );
    await act(async () => {
      [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === 'open')!.click();
      await vi.waitFor(() => expect(library.getLibraryWorkflowRecordCached).toHaveBeenCalled());
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      });
    });
  };

  beforeEach(() => {
    projectId = 'project-1';
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    commands = { addWorkflow: vi.fn(), replaceWorkflow: vi.fn() };
    notifications = { error: vi.fn(), info: vi.fn() };
    onOpened.mockClear();
    library.beforeAnswer = () => {};
    library.getLibraryWorkflowRecordCached.mockReset().mockResolvedValue(RECORD);
    workflowFitViewRequestStore.setSnapshot({ request: null });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('puts the library version in the chosen copy at its revision and fits the copy’s kept editor', async () => {
    await runReplace();

    expect(commands.replaceWorkflow).toHaveBeenCalledExactlyOnceWith(
      { projectId: 'project-1', workflowId: 'copy-1' },
      expect.objectContaining({ nodes: [expect.objectContaining({ id: 'value' })] }),
      {
        label: 'workflowLibrary.replacedLabel',
        source: { libraryWorkflowId: ITEM.workflow_id, ownerUserId: 'user-a', revision: 4 },
      }
    );
    expect(commands.addWorkflow).not.toHaveBeenCalled();
    expect(workflowFitViewRequestStore.getSnapshot().request).toMatchObject({
      nodes: [{ id: 'value', position: { x: 400, y: 300 } }],
      scope: { projectId: 'project-1', workflowId: 'copy-1' },
    });
    expect(onOpened).toHaveBeenCalledTimes(1);
  });

  it('replaces nothing when another project became active while the template loaded', async () => {
    library.beforeAnswer = () => {
      projectId = 'project-2';
    };

    await runReplace();

    expect(commands.replaceWorkflow).not.toHaveBeenCalled();
    expect(notifications.info).toHaveBeenCalledWith('workflowLibrary.openProjectChanged');
    expect(onOpened).not.toHaveBeenCalled();
  });

  it('opens another account’s public workflow for an admin without granting update access', async () => {
    identity.session = {
      multiuserEnabled: true,
      phase: 'ready',
      user: { is_admin: true, user_id: 'admin-user' },
    };
    library.getLibraryWorkflowRecordCached.mockResolvedValue({
      ...RECORD,
      is_public: true,
      user_id: 'owner-user',
    });

    await runOpen();

    expect(commands.addWorkflow).toHaveBeenCalledTimes(1);
    expect(commands.addWorkflow).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        source: { libraryWorkflowId: ITEM.workflow_id, ownerUserId: 'owner-user', revision: 4 },
      })
    );
    expect(commands.replaceWorkflow).not.toHaveBeenCalled();
    expect(onOpened).toHaveBeenCalledTimes(1);
  });

  it('keeps the update hint for a current owner of a private Library workflow', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'owner-user' } };
    library.getLibraryWorkflowRecordCached.mockResolvedValue({
      ...RECORD,
      is_public: false,
      user_id: 'owner-user',
    });

    await runOpen();

    expect(commands.addWorkflow).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        source: { libraryWorkflowId: ITEM.workflow_id, ownerUserId: 'owner-user', revision: 4 },
      })
    );
  });

  it('does not attach owner metadata to a bundled default workflow', async () => {
    library.getLibraryWorkflowRecordCached.mockResolvedValue({
      ...RECORD,
      is_public: true,
      user_id: 'system',
      workflow: { ...RECORD.workflow, meta: { category: 'default', version: '3.0.0' } },
    });

    await runOpen();

    expect(commands.addWorkflow.mock.calls[0]?.[1]?.source).toEqual({
      libraryWorkflowId: ITEM.workflow_id,
      revision: 4,
    });
  });

  it('withholds the update hint for a multiuser Library record without a verified owner', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'current-user' } };
    library.getLibraryWorkflowRecordCached.mockResolvedValue({ ...RECORD, is_public: true, user_id: undefined });

    await runOpen();

    expect(commands.addWorkflow).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        source: { libraryWorkflowId: ITEM.workflow_id, revision: 4 },
      })
    );
  });
});
