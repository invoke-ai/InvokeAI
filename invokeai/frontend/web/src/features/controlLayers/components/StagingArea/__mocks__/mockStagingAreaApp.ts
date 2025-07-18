import { merge } from 'es-toolkit';
import type { StagingAreaAppApi } from 'features/controlLayers/components/StagingArea/state';
import type { AutoSwitchMode } from 'features/controlLayers/store/canvasSettingsSlice';
import type { ImageDTO, S } from 'services/api/types';
import type { PartialDeep } from 'type-fest';
import { vi } from 'vitest';

export const createMockStagingAreaApp = (): StagingAreaAppApi & {
  // Additional methods for testing
  _triggerItemsChanged: (items: S['SessionQueueItem'][]) => void;
  _triggerQueueItemStatusChanged: (data: S['QueueItemStatusChangedEvent']) => void;
  _triggerInvocationProgress: (data: S['InvocationProgressEvent']) => void;
  _setAutoSwitchMode: (mode: AutoSwitchMode) => void;
  _setImageDTO: (imageName: string, imageDTO: ImageDTO | null) => void;
  _setLoadImageDelay: (delay: number) => void;
} => {
  const itemsChangedHandlers = new Set<(items: S['SessionQueueItem'][]) => void>();
  const queueItemStatusChangedHandlers = new Set<(data: S['QueueItemStatusChangedEvent']) => void>();
  const invocationProgressHandlers = new Set<(data: S['InvocationProgressEvent']) => void>();

  let autoSwitchMode: AutoSwitchMode = 'switch_on_start';
  const imageDTOs = new Map<string, ImageDTO | null>();
  let loadImageDelay = 0;

  return {
    onDiscard: vi.fn(),
    onDiscardAll: vi.fn(),
    onAccept: vi.fn(),
    onSelect: vi.fn(),
    onSelectPrev: vi.fn(),
    onSelectNext: vi.fn(),
    onSelectFirst: vi.fn(),
    onSelectLast: vi.fn(),
    getAutoSwitch: vi.fn(() => autoSwitchMode),
    onAutoSwitchChange: vi.fn(),
    getImageDTO: vi.fn((imageName: string) => {
      return Promise.resolve(imageDTOs.get(imageName) || null);
    }),
    loadImage: vi.fn(async (imageName: string) => {
      if (loadImageDelay > 0) {
        await new Promise((resolve) => {
          setTimeout(resolve, loadImageDelay);
        });
      }
      // Mock HTMLImageElement for testing environment
      const mockImage = {
        src: imageName,
        width: 512,
        height: 512,
        onload: null,
        onerror: null,
      } as HTMLImageElement;
      return mockImage;
    }),
    onItemsChanged: vi.fn((handler) => {
      itemsChangedHandlers.add(handler);
      return () => itemsChangedHandlers.delete(handler);
    }),
    onQueueItemStatusChanged: vi.fn((handler) => {
      queueItemStatusChangedHandlers.add(handler);
      return () => queueItemStatusChangedHandlers.delete(handler);
    }),
    onInvocationProgress: vi.fn((handler) => {
      invocationProgressHandlers.add(handler);
      return () => invocationProgressHandlers.delete(handler);
    }),

    // Testing helper methods
    _triggerItemsChanged: (items: S['SessionQueueItem'][]) => {
      itemsChangedHandlers.forEach((handler) => handler(items));
    },
    _triggerQueueItemStatusChanged: (data: S['QueueItemStatusChangedEvent']) => {
      queueItemStatusChangedHandlers.forEach((handler) => handler(data));
    },
    _triggerInvocationProgress: (data: S['InvocationProgressEvent']) => {
      invocationProgressHandlers.forEach((handler) => handler(data));
    },
    _setAutoSwitchMode: (mode: AutoSwitchMode) => {
      autoSwitchMode = mode;
    },
    _setImageDTO: (imageName: string, imageDTO: ImageDTO | null) => {
      imageDTOs.set(imageName, imageDTO);
    },
    _setLoadImageDelay: (delay: number) => {
      loadImageDelay = delay;
    },
  };
};

export const createMockQueueItem = (overrides: PartialDeep<S['SessionQueueItem']> = {}): S['SessionQueueItem'] =>
  merge(
    {
      item_id: 1,
      batch_id: 'test-batch-id',
      session_id: 'test-session',
      queue_id: 'test-queue-id',
      status: 'pending',
      priority: 0,
      origin: null,
      destination: 'test-session',
      error_type: null,
      error_message: null,
      error_traceback: null,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      started_at: null,
      completed_at: null,
      field_values: null,
      retried_from_item_id: null,
      is_api_validation_run: false,
      published_workflow_id: null,
      session: {
        id: 'test-session',
        graph: {},
        execution_graph: {},
        executed: [],
        executed_history: [],
        results: {
          'test-node-id': {
            image: {
              image_name: 'test-image.png',
            },
          },
        },
        errors: {},
        prepared_source_mapping: {},
        source_prepared_mapping: {
          canvas_output: ['test-node-id'],
        },
      },
      workflow: null,
    },
    overrides
  ) as S['SessionQueueItem'];

export const createMockImageDTO = (overrides: Partial<ImageDTO> = {}): ImageDTO => ({
  image_name: 'test-image.png',
  image_url: 'http://test.com/test-image.png',
  thumbnail_url: 'http://test.com/test-image-thumb.png',
  image_origin: 'internal',
  image_category: 'general',
  width: 512,
  height: 512,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  deleted_at: null,
  is_intermediate: false,
  starred: false,
  has_workflow: false,
  session_id: 'test-session',
  node_id: 'test-node-id',
  board_id: null,
  ...overrides,
});

export const createMockProgressEvent = (
  overrides: PartialDeep<S['InvocationProgressEvent']> = {}
): S['InvocationProgressEvent'] =>
  merge(
    {
      timestamp: Date.now(),
      queue_id: 'test-queue-id',
      item_id: 1,
      batch_id: 'test-batch-id',
      session_id: 'test-session',
      origin: null,
      destination: 'test-session',
      invocation: {},
      invocation_source_id: 'test-invocation-source-id',
      message: 'Processing...',
      percentage: 50,
      image: null,
    } as S['InvocationProgressEvent'],
    overrides
  );

export const createMockQueueItemStatusChangedEvent = (
  overrides: PartialDeep<S['QueueItemStatusChangedEvent']> = {}
): S['QueueItemStatusChangedEvent'] =>
  merge(
    {
      timestamp: Date.now(),
      queue_id: 'test-queue-id',
      item_id: 1,
      batch_id: 'test-batch-id',
      origin: null,
      destination: 'test-session',
      status: 'completed',
      error_type: null,
      error_message: null,
    } as S['QueueItemStatusChangedEvent'],
    overrides
  );
