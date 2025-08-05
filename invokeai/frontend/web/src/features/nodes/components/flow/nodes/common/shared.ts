// Certain CSS transitions are disabled as a performance optimization - they can cause massive slowdowns in large
// workflows even when the animations are GPU-accelerated CSS.

import type { SystemStyleObject } from '@invoke-ai/ui-library';

export const containerSx: SystemStyleObject = {
  h: 'full',
  position: 'relative',
  borderRadius: 'base',
  transitionProperty: 'none',
  cursor: 'grab',
  '--border-color': 'var(--invoke-colors-base-500)',
  '--border-color-selected': 'var(--invoke-colors-blue-300)',
  '--header-bg-color': 'var(--invoke-colors-base-900)',
  '&[data-status="warning"]': {
    '--border-color': 'var(--invoke-colors-warning-500)',
    '--border-color-selected': 'var(--invoke-colors-warning-500)',
    '--header-bg-color': 'var(--invoke-colors-warning-700)',
  },
  '&[data-status="error"]': {
    '--border-color': 'var(--invoke-colors-error-500)',
    '--border-color-selected': 'var(--invoke-colors-error-500)',
    '--header-bg-color': 'var(--invoke-colors-error-700)',
  },
  // The action buttons are hidden by default and shown on hover
  '& .node-selection-overlay': {
    display: 'block',
    position: 'absolute',
    top: 0,
    insetInlineEnd: 0,
    bottom: 0,
    insetInlineStart: 0,
    borderRadius: 'base',
    transitionProperty: 'none',
    pointerEvents: 'none',
    shadow: '0 0 0 1px var(--border-color)',
  },
  '&[data-is-mouse-over-node="true"] .node-selection-overlay': {
    display: 'block',
  },
  '&[data-is-mouse-over-form-field="true"] .node-selection-overlay': {
    display: 'block',
    bg: 'invokeBlueAlpha.100',
  },
  _hover: {
    '& .node-selection-overlay': {
      display: 'block',
      shadow: '0 0 0 1px var(--border-color-selected)',
    },
    '&[data-is-selected="true"] .node-selection-overlay': {
      display: 'block',
      shadow: '0 0 0 2px var(--border-color-selected)',
    },
  },
  '&[data-is-selected="true"] .node-selection-overlay': {
    display: 'block',
    shadow: '0 0 0 2px var(--border-color-selected)',
  },
  '&[data-is-editor-locked="true"]': {
    '& *': {
      cursor: 'not-allowed',
      pointerEvents: 'none',
    },
  },
};

export const shadowsSx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'base',
  pointerEvents: 'none',
  zIndex: -1,
  shadow: 'var(--invoke-shadows-xl), var(--invoke-shadows-base), var(--invoke-shadows-base)',
};

export const inProgressSx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  insetInlineEnd: 0,
  bottom: 0,
  insetInlineStart: 0,
  borderRadius: 'md',
  pointerEvents: 'none',
  transitionProperty: 'none',
  opacity: 0.7,
  zIndex: -1,
  display: 'none',
  shadow: '0 0 0 2px var(--invoke-colors-yellow-400), 0 0 20px 2px var(--invoke-colors-orange-700)',
  '&[data-is-in-progress="true"]': {
    display: 'block',
  },
};
