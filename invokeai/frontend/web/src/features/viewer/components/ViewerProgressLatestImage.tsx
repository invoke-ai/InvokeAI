import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import type { TypesafeDraggableData } from 'features/dnd/types';
import { latestLinearImageLoaded } from 'features/progress/store/progressSlice';
import { memo, useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const ViewerProgressLatestImage = memo(({ imageDTO }: Props) => {
  const dispatch = useAppDispatch();
  const progressImageDataURL = useAppSelector((s) => s.progress.linearDenoiseProgress?.progress_image?.dataURL);
  const draggableData = useMemo<TypesafeDraggableData | undefined>(
    () => ({
      id: 'current-image',
      payloadType: 'IMAGE_DTO',
      payload: { imageDTO },
    }),
    [imageDTO]
  );

  const onLoad = useCallback(() => {
    dispatch(latestLinearImageLoaded());
  }, [dispatch]);

  return (
    <IAIDndImage
      imageDTO={imageDTO}
      draggableData={draggableData}
      isUploadDisabled={true}
      fitContainer
      fallbackSrc={progressImageDataURL}
      dataTestId="progress-latest-image"
      onLoad={onLoad}
    />
  );
});

ViewerProgressLatestImage.displayName = 'ViewerProgressLatestImage';
