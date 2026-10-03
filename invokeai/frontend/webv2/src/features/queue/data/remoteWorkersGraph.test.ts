import type { QueueBackendGraph } from '@features/queue/core/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  getRemoteWorkerName: vi.fn(),
  getRemoteWorkerUrls: vi.fn(),
  getRemoteWorkersSettings: vi.fn(),
  isRemoteWorkerEnabled: vi.fn(),
}));

vi.mock('./remoteWorkersStore', () => ({
  getRemoteWorkerName: mocks.getRemoteWorkerName,
  getRemoteWorkerUrls: mocks.getRemoteWorkerUrls,
  getRemoteWorkersSettings: mocks.getRemoteWorkersSettings,
  isRemoteWorkerEnabled: mocks.isRemoteWorkerEnabled,
}));

import { applyRemoteWorkersToGraph } from './remoteWorkersGraph';
import { AUTOMATIC_REMOTE_WORKER_NODE_ID, AUTOMATIC_REMOTE_WORKER_NODE_TYPE } from './remoteWorkersGraphContract';

const worker1 = 'http://worker-1:9090';
const worker2 = 'http://worker-2:9090';

const makeGraph = (): QueueBackendGraph => ({
  id: 'graph-1',
  edges: [
    {
      source: { node_id: 'prompt', field: 'text' },
      destination: { node_id: 'denoise', field: 'positive_conditioning' },
    },
  ],
  nodes: {
    prompt: { id: 'prompt', type: 'string', text: 'hello' },
    denoise: { id: 'denoise', type: 'denoise' },
  },
});

describe('applyRemoteWorkersToGraph', () => {
  beforeEach(() => {
    mocks.getRemoteWorkersSettings.mockReturnValue({
      enabled: true,
      workerUrls: `${worker1}\n${worker2}`,
      workerNames: {},
      disabledWorkerUrls: [],
      dispatchMode: 'distributed',
      autoTransferMissingModels: true,
      modelTransferHost: '  primary-host  ',
      keepRemoteCopies: false,
    });
    mocks.getRemoteWorkerUrls.mockReturnValue([worker1, worker2]);
    mocks.isRemoteWorkerEnabled.mockReturnValue(true);
    mocks.getRemoteWorkerName.mockImplementation((_url: string, index: number) =>
      index === 0 ? 'RTX5080' : 'Server GPU'
    );
  });

  it('leaves the graph untouched when Remote Workers are disabled', () => {
    const graph = makeGraph();
    mocks.getRemoteWorkersSettings.mockReturnValue({ enabled: false });

    expect(applyRemoteWorkersToGraph(graph, null, 'gallery')).toBe(graph);
  });

  it('does not overwrite a pre-existing node that uses the automatic helper id', () => {
    const graph = makeGraph();
    graph.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID] = {
      id: AUTOMATIC_REMOTE_WORKER_NODE_ID,
      type: 'some_other_node',
    };

    expect(applyRemoteWorkersToGraph(graph, null, 'gallery')).toBe(graph);
  });

  it('injects one Distributed helper while preserving the executable graph', () => {
    const graph = makeGraph();
    const result = applyRemoteWorkersToGraph(graph, 'board-1', 'gallery');
    const helper = result.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID];

    expect(result).not.toBe(graph);
    expect(result.edges).toEqual(graph.edges);
    expect(result.nodes.prompt).toEqual(graph.nodes.prompt);
    expect(helper).toMatchObject({
      id: AUTOMATIC_REMOTE_WORKER_NODE_ID,
      type: AUTOMATIC_REMOTE_WORKER_NODE_TYPE,
      remote_url: worker1,
      additional_remote_urls: worker2,
      remote_worker_names: JSON.stringify(['RTX5080', 'Server GPU']),
      dispatch_mode: 'Distributed',
      auto_transfer_missing_models: true,
      model_transfer_host: 'primary-host',
      keep_remote_copies: false,
      result_destination: 'gallery',
      local_gallery_board_id: 'board-1',
    });
    expect(helper).not.toHaveProperty('source_graph_json');
  });

  it('keeps the executable graph intact for Remote Only', () => {
    const graph = makeGraph();
    mocks.getRemoteWorkersSettings.mockReturnValue({
      enabled: true,
      workerUrls: `${worker1}\n${worker2}`,
      workerNames: {},
      disabledWorkerUrls: [],
      dispatchMode: 'remote_only',
      autoTransferMissingModels: false,
      modelTransferHost: '',
      keepRemoteCopies: true,
    });

    const result = applyRemoteWorkersToGraph(graph, 'board-1', 'gallery');
    const helper = result.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID];

    expect(result.edges).toEqual(graph.edges);
    expect(result.nodes.prompt).toEqual(graph.nodes.prompt);
    expect(helper).toMatchObject({
      dispatch_mode: 'Remote Only',
      result_destination: 'gallery',
      local_gallery_board_id: 'board-1',
      keep_remote_copies: true,
    });
    expect(helper).not.toHaveProperty('source_graph_json');
  });

  it('never carries a Gallery board into a Canvas dispatch', () => {
    const graph = makeGraph();
    const result = applyRemoteWorkersToGraph(graph, 'board-1', 'canvas');

    expect(result.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID]?.local_gallery_board_id).toBe('');
    expect(result.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID]?.result_destination).toBe('canvas');
  });

  it('excludes workers disabled with the power toggle without consulting browser health', () => {
    const graph = makeGraph();
    mocks.isRemoteWorkerEnabled.mockImplementation((url: string) => url === worker2);

    const result = applyRemoteWorkersToGraph(graph, null, 'gallery');
    const helper = result.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID];

    expect(helper).toMatchObject({
      remote_url: worker2,
      additional_remote_urls: '',
      remote_worker_names: JSON.stringify(['Server GPU']),
    });
  });

  it('leaves Distributed local-only when every configured worker is disabled', () => {
    const graph = makeGraph();
    mocks.isRemoteWorkerEnabled.mockReturnValue(false);

    expect(applyRemoteWorkersToGraph(graph, null, 'gallery')).toBe(graph);
  });

  it('keeps a Remote Only helper when every configured worker is disabled', () => {
    const graph = makeGraph();
    mocks.getRemoteWorkersSettings.mockReturnValue({
      enabled: true,
      workerUrls: `${worker1}\n${worker2}`,
      workerNames: {},
      disabledWorkerUrls: [worker1, worker2],
      dispatchMode: 'remote_only',
      autoTransferMissingModels: false,
      modelTransferHost: '',
      keepRemoteCopies: false,
    });
    mocks.isRemoteWorkerEnabled.mockReturnValue(false);

    const result = applyRemoteWorkersToGraph(graph, null, 'gallery');
    const helper = result.nodes[AUTOMATIC_REMOTE_WORKER_NODE_ID];

    expect(result).not.toBe(graph);
    expect(helper).toMatchObject({
      dispatch_mode: 'Remote Only',
      additional_remote_urls: '',
      remote_worker_names: '[]',
    });
    expect(helper?.remote_url).toBeUndefined();
  });
});
