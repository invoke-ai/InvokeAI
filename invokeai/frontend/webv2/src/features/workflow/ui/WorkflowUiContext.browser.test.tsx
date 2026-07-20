import type { ReactNode } from 'react';

import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkflowGraphPreviewPort, WorkflowReadPort, WorkflowUiAdapter } from './WorkflowUiContext';

import {
  useWorkflowCapabilitiesSelector,
  useWorkflowGraphPreview,
  useWorkflowPreferencesSelector,
  useWorkflowProjectSelector,
  useWorkflowUi,
  WorkflowGraphPreviewProvider,
  WorkflowUiProvider,
} from './WorkflowUiContext';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  const port: WorkflowReadPort<Snapshot> = {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    port,
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

const projectState = (id = 'project-1') => ({
  galleryValues: {},
  graphHistory: [],
  id,
  isWorkflowRunning: false,
  projectGraph: { edges: [], nodes: [], version: 1 as const },
  workflowValues: {},
});

const preferencesState = () => ({
  reduceMotion: false,
  themeId: 'classic' as const,
  workflowEdgeStyle: 'curved' as const,
  workflowShowMinimap: true,
  workflowSnapToGrid: false,
  workflowValidateConnections: true,
});

describe('Workflow UI read-port isolation', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('updates only consumers of the changed project, preference, capability, or graph-preview port', async () => {
    const project = createMutablePort(projectState());
    const preferences = createMutablePort(preferencesState());
    const capabilities = createMutablePort({ canUseCache: true });
    const counts = { capability: 0, graph: 0, preferences: 0, project: 0, projectEqual: 0, services: 0 };
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally stable for this render lifetime
    const adapter = {
      capabilities: capabilities.port,
      commands: {},
      getProjectGraph: () => project.port.getSnapshot().projectGraph,
      nodeExecution: { get: () => null, subscribe: () => vi.fn() },
      notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
      performance: { mark: vi.fn(), measure: vi.fn(), time: vi.fn() },
      preferences: preferences.port,
      project: project.port,
      registerModalHotkeyLayer: vi.fn(() => vi.fn()),
      widgets: {},
    } as unknown as WorkflowUiAdapter;

    let setGraphPreview: ((adapter: WorkflowGraphPreviewPort) => void) | null = null;
    const initialGraphPreview: WorkflowGraphPreviewPort = {
      getRoute: () => null,
      invoke: () => Promise.resolve(false),
    };
    const GraphPreviewHarness = ({ children }: { children: ReactNode }) => {
      const [graphPreview, setGraphPreviewState] = useState(initialGraphPreview);
      setGraphPreview = setGraphPreviewState;
      return <WorkflowGraphPreviewProvider adapter={graphPreview}>{children}</WorkflowGraphPreviewProvider>;
    };
    const ProjectConsumer = () => {
      useWorkflowProjectSelector((snapshot) => snapshot.projectGraph);
      counts.project += 1;
      return null;
    };
    const EqualProjectConsumer = () => {
      useWorkflowProjectSelector(
        (snapshot) => [snapshot.id],
        (left, right) => left[0] === right[0]
      );
      counts.projectEqual += 1;
      return null;
    };
    const PreferencesConsumer = () => {
      useWorkflowPreferencesSelector((snapshot) => snapshot.workflowSnapToGrid);
      counts.preferences += 1;
      return null;
    };
    const CapabilityConsumer = () => {
      useWorkflowCapabilitiesSelector((snapshot) => snapshot.canUseCache);
      counts.capability += 1;
      return null;
    };
    const ServicesConsumer = () => {
      useWorkflowUi();
      counts.services += 1;
      return null;
    };
    const GraphConsumer = () => {
      useWorkflowGraphPreview();
      counts.graph += 1;
      return null;
    };

    await act(() => {
      root.render(
        <WorkflowUiProvider adapter={adapter}>
          <GraphPreviewHarness>
            <ProjectConsumer />
            <EqualProjectConsumer />
            <PreferencesConsumer />
            <CapabilityConsumer />
            <ServicesConsumer />
            <GraphConsumer />
          </GraphPreviewHarness>
        </WorkflowUiProvider>
      );
    });
    expect(counts).toEqual({ capability: 1, graph: 1, preferences: 1, project: 1, projectEqual: 1, services: 1 });

    await act(() => project.setSnapshot({ ...projectState(), projectGraph: { edges: [], nodes: [], version: 1 } }));
    expect(counts).toEqual({ capability: 1, graph: 1, preferences: 1, project: 2, projectEqual: 1, services: 1 });

    await act(() => preferences.setSnapshot({ ...preferencesState(), workflowSnapToGrid: true }));
    expect(counts).toEqual({ capability: 1, graph: 1, preferences: 2, project: 2, projectEqual: 1, services: 1 });

    await act(() => capabilities.setSnapshot({ canUseCache: false }));
    expect(counts).toEqual({ capability: 2, graph: 1, preferences: 2, project: 2, projectEqual: 1, services: 1 });

    await act(() => setGraphPreview?.({ getRoute: () => null, invoke: () => Promise.resolve(true) }));
    await act(() => setGraphPreview?.({ getRoute: () => null, invoke: () => Promise.resolve(false) }));
    expect(counts).toEqual({ capability: 2, graph: 3, preferences: 2, project: 2, projectEqual: 1, services: 1 });
  });
});
