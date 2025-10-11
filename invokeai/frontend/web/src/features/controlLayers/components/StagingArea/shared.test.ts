import type { S } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { getOutputImageName, getProgressMessage, getQueueItemElementId } from './shared';

describe('StagingAreaApi Utility Functions', () => {
  describe('getProgressMessage', () => {
    it('should return default message when no data provided', () => {
      expect(getProgressMessage()).toBe('Generating');
      expect(getProgressMessage(null)).toBe('Generating');
    });

    it('should format progress message when data is provided', () => {
      const progressEvent: S['InvocationProgressEvent'] = {
        item_id: 1,
        destination: 'test-session',
        node_id: 'test-node',
        source_node_id: 'test-source-node',
        progress: 0.5,
        message: 'Processing image...',
        image: null,
      } as unknown as S['InvocationProgressEvent'];

      const result = getProgressMessage(progressEvent);
      expect(result).toBe('Processing image...');
    });
  });

  describe('getQueueItemElementId', () => {
    it('should generate correct element ID for queue item', () => {
      expect(getQueueItemElementId(0)).toBe('queue-item-preview-0');
      expect(getQueueItemElementId(5)).toBe('queue-item-preview-5');
      expect(getQueueItemElementId(99)).toBe('queue-item-preview-99');
    });
  });

  describe('getOutputImageName', () => {
    it('should extract image name from completed queue item', () => {
      const queueItem: S['SessionQueueItem'] = {
        item_id: 1,
        status: 'completed',
        priority: 0,
        destination: 'test-session',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        started_at: '2024-01-01T00:00:01Z',
        completed_at: '2024-01-01T00:01:00Z',
        error: null,
        session: {
          id: 'test-session',
          source_prepared_mapping: {
            canvas_output: ['output-node-id'],
          },
          results: {
            'output-node-id': {
              image: {
                image_name: 'test-output.png',
              },
            },
          },
        },
      } as unknown as S['SessionQueueItem'];

      expect(getOutputImageName(queueItem)).toBe('test-output.png');
    });

    it('should use fallback when no canvas output node found', () => {
      const queueItem = {
        item_id: 1,
        status: 'completed',
        priority: 0,
        destination: 'test-session',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        started_at: '2024-01-01T00:00:01Z',
        completed_at: '2024-01-01T00:01:00Z',
        error: null,
        session: {
          id: 'test-session',
          source_prepared_mapping: {
            some_other_node: ['other-node-id'],
          },
          results: {
            'other-node-id': {
              type: 'image_output',
              image: {
                image_name: 'test-output.png',
              },
              width: 512,
              height: 512,
            },
          },
        },
      } as unknown as S['SessionQueueItem'];

      // Fallback mechanism finds image in other nodes when no canvas_output node exists
      expect(getOutputImageName(queueItem)).toBe('test-output.png');
    });

    it('should return null when output node has no results', () => {
      const queueItem: S['SessionQueueItem'] = {
        item_id: 1,
        status: 'completed',
        priority: 0,
        destination: 'test-session',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        started_at: '2024-01-01T00:00:01Z',
        completed_at: '2024-01-01T00:01:00Z',
        error: null,
        session: {
          id: 'test-session',
          source_prepared_mapping: {
            canvas_output: ['output-node-id'],
          },
          results: {},
        },
      } as unknown as S['SessionQueueItem'];

      expect(getOutputImageName(queueItem)).toBe(null);
    });

    it('should return null when results contain no image fields', () => {
      const queueItem: S['SessionQueueItem'] = {
        item_id: 1,
        status: 'completed',
        priority: 0,
        destination: 'test-session',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        started_at: '2024-01-01T00:00:01Z',
        completed_at: '2024-01-01T00:01:00Z',
        error: null,
        session: {
          id: 'test-session',
          source_prepared_mapping: {
            canvas_output: ['output-node-id'],
          },
          results: {
            'output-node-id': {
              text: 'some text output',
              number: 42,
            },
          },
        },
      } as unknown as S['SessionQueueItem'];

      expect(getOutputImageName(queueItem)).toBe(null);
    });

    it('should handle multiple outputs and return first image', () => {
      const queueItem: S['SessionQueueItem'] = {
        item_id: 1,
        status: 'completed',
        priority: 0,
        destination: 'test-session',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        started_at: '2024-01-01T00:00:01Z',
        completed_at: '2024-01-01T00:01:00Z',
        error: null,
        session: {
          id: 'test-session',
          source_prepared_mapping: {
            canvas_output: ['output-node-id'],
          },
          results: {
            'output-node-id': {
              text: 'some text',
              first_image: {
                image_name: 'first-image.png',
              },
              second_image: {
                image_name: 'second-image.png',
              },
            },
          },
        },
      } as unknown as S['SessionQueueItem'];

      const result = getOutputImageName(queueItem);
      expect(result).toBe('first-image.png');
    });

    it('should handle empty session mapping', () => {
      const queueItem: S['SessionQueueItem'] = {
        item_id: 1,
        status: 'completed',
        priority: 0,
        destination: 'test-session',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        started_at: '2024-01-01T00:00:01Z',
        completed_at: '2024-01-01T00:01:00Z',
        error: null,
        session: {
          id: 'test-session',
          source_prepared_mapping: {},
          results: {},
        },
      } as unknown as S['SessionQueueItem'];

      expect(getOutputImageName(queueItem)).toBe(null);
    });
  });
});
