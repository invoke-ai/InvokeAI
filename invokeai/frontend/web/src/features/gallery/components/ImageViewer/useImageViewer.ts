import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useMemo } from 'react';

const hook = buildUseBoolean(true);
const useImageViewerState = hook[0];

export const $imageViewerState = hook[1];

export const useImageViewer = () => {
  const imageViewerState = useImageViewerState();
  const isOpen = useMemo(() => imageViewerState.isTrue, [imageViewerState]);
  const open = useMemo(() => imageViewerState.setTrue, [imageViewerState]);
  const close = useMemo(() => imageViewerState.setFalse, [imageViewerState]);
  const toggle = useMemo(() => imageViewerState.toggle, [imageViewerState]);

  return { isOpen, open, close, toggle, $state: $imageViewerState };
};
