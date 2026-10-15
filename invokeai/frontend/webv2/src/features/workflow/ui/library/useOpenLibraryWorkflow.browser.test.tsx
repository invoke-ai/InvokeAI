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
  name: 'Portrait',
  revision: 4,
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
    const { replace } = useOpenLibraryWorkflow(onOpened);

    return (
      <button type="button" onClick={() => void replace(ITEM, 'copy-1')}>
        replace
      </button>
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
      host.querySelector('button')!.click();
      await vi.waitFor(() => expect(library.getLibraryWorkflowRecordCached).toHaveBeenCalled());
      // The load waits two frames for the busy overlay to paint before applying.
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      });
    });
  };

  beforeEach(() => {
    projectId = 'project-1';
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
      { label: 'workflowLibrary.replacedLabel', source: { libraryWorkflowId: ITEM.workflow_id, revision: 4 } }
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
});
