import type { SystemStyleObject } from '@invoke-ai/ui-library';
import type { CSSProperties } from 'react';

/** Inner chrome for React Flow handles on invocation ports and connector passthrough (colors via Box props). */
export const NODE_IO_HANDLE_INNER_SX = {
  position: 'relative',
  width: 'full',
  height: 'full',
  borderStyle: 'solid',
  borderWidth: 4,
  pointerEvents: 'none',
  '&[data-cardinality="SINGLE"]': {
    borderWidth: 0,
  },
  borderRadius: '100%',
  '&[data-is-model-field="true"], &[data-is-batch-field="true"]': {
    borderRadius: 4,
  },
  '&[data-is-batch-field="true"]': {
    transform: 'rotate(45deg)',
  },
  '&[data-is-connection-in-progress="true"][data-is-connection-start-field="false"][data-is-connection-valid="false"]':
    {
      filter: 'opacity(0.4) grayscale(0.7)',
      cursor: 'not-allowed',
    },
  '&[data-is-connection-in-progress="true"][data-is-connection-start-field="true"][data-is-connection-valid="false"]': {
    cursor: 'grab',
  },
  '&[data-is-connection-in-progress="false"] &[data-is-connection-valid="true"]': {
    cursor: 'crosshair',
  },
} satisfies SystemStyleObject;

/** Target side (`Position.Left`). Matches InputFieldHandle hit area. */
export const NODE_IO_HANDLE_HITBOX_INPUT: CSSProperties = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  zIndex: 1,
  background: 'none',
  border: 'none',
  insetInlineStart: '-0.5rem',
};

/** Source side (`Position.Right`). Matches OutputFieldHandle hit area. */
export const NODE_IO_HANDLE_HITBOX_OUTPUT: CSSProperties = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  zIndex: 1,
  background: 'none',
  border: 'none',
  insetInlineEnd: '-0.5rem',
};
