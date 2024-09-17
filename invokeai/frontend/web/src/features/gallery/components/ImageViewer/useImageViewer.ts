import { useAppDispatch } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { imageSelected, imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

const [useImageViewerState, $imageViewerState] = buildUseBoolean(true);

export const useImageViewer = () => {
  const dispatch = useAppDispatch();
  const imageViewerState = useImageViewerState();
  const isOpen = useMemo(() => imageViewerState.isTrue, [imageViewerState]);
  const open = useMemo(() => imageViewerState.setTrue, [imageViewerState]);
  const close = useMemo(() => imageViewerState.setFalse, [imageViewerState]);
  const toggle = useMemo(() => imageViewerState.toggle, [imageViewerState]);
  const openImageInViewer = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(imageToCompareChanged(null));
      dispatch(imageSelected(imageDTO));
      open();
    },
    [dispatch, open]
  );

  return { isOpen, open, close, toggle, $state: $imageViewerState, openImageInViewer };
};
