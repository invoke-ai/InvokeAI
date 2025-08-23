import type { SystemStyleObject } from '@invoke-ai/ui-library';

export const galleryItemContainerSX = {
  containerType: 'inline-size',
  w: 'full',
  h: 'full',
  '.gallery-image-size-badge': {
    '@container (max-width: 80px)': {
      '&': { display: 'none' },
    },
  },
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
  userSelect: 'none',
  webkitUserSelect: 'none',
  position: 'relative',
  justifyContent: 'center',
  alignItems: 'center',
  aspectRatio: '1/1',
  '::before': {
    content: '""',
    display: 'inline-block',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
    borderRadius: 'base',
  },
  '&[data-selected=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-500), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
  },
  '&[data-selected-for-compare=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-300), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
  },
  '&:hover::before': {
    boxShadow:
      'inset 0px 0px 0px 1px var(--invoke-colors-invokeBlue-300), inset 0px 0px 0px 2px var(--invoke-colors-invokeBlue-800)',
  },
  '&:hover[data-selected=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-400), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
  },
  '&:hover[data-selected-for-compare=true]::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-200), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
  },
} satisfies SystemStyleObject;
