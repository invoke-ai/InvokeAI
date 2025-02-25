import type { SystemStyleObject } from '@invoke-ai/ui-library';

// This must be in this file to avoid circular dependencies
export const getEditModeWrapperId = (id: string) => `${id}-edit-mode-wrapper`;

export const formElementDndSx: SystemStyleObject = {
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
  '&[data-active-drop-region="center"]': {
    opacity: 1,
    bg: 'base.850',
  },
};

export const formElementIsDraggingSx: SystemStyleObject = {
  opacity: 0.3,
};

export const formElementIsActiveDropRegionSx: SystemStyleObject = {
  opacity: 1,
  bg: 'base.850',
};
