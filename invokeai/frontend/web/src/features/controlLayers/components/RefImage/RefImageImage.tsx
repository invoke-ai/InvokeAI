import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { bboxSizeOptimized, bboxSizeRecalled } from 'features/controlLayers/store/canvasSlice';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { sizeOptimized, sizeRecalled } from 'features/controlLayers/store/paramsSlice';
import type { ImageWithDims } from 'features/controlLayers/store/types';
import type { setGlobalReferenceImageDndTarget, setRegionalGuidanceReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiRulerBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

type Props<T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget> = {
  image: ImageWithDims | null;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  dndTarget: T;
  dndTargetData: ReturnType<T['getData']>;
};

export const RefImageImage = memo(
  <T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget>({
    image,
    onChangeImage,
    dndTarget,
    dndTargetData,
  }: Props<T>) => {
    const { t } = useTranslation();
    const store = useAppStore();
    const isConnected = useStore($isConnected);
    const tab = useAppSelector(selectActiveTab);
    const isStaging = useCanvasIsStaging();
    const { currentData: imageDTO, isError } = useGetImageDTOQuery(image?.image_name ?? skipToken);
    const handleResetControlImage = useCallback(() => {
      onChangeImage(null);
    }, [onChangeImage]);

    useEffect(() => {
      if (isConnected && isError) {
        handleResetControlImage();
      }
    }, [handleResetControlImage, isError, isConnected]);

    const onUpload = useCallback(
      (imageDTO: ImageDTO) => {
        onChangeImage(imageDTO);
      },
      [onChangeImage]
    );

    const recallSizeAndOptimize = useCallback(() => {
      if (!imageDTO || (tab === 'canvas' && isStaging)) {
        return;
      }
      const { width, height } = imageDTO;
      if (tab === 'canvas') {
        store.dispatch(bboxSizeRecalled({ width, height }));
        store.dispatch(bboxSizeOptimized());
      } else if (tab === 'generate') {
        store.dispatch(sizeRecalled({ width, height }));
        store.dispatch(sizeOptimized());
      }
    }, [imageDTO, isStaging, store, tab]);

    return (
      <Flex position="relative" w="full" h="full" alignItems="center" data-error={!imageDTO && !image?.image_name}>
        {!imageDTO && (
          <UploadImageIconButton
            w="full"
            h="full"
            isError={!imageDTO && !image?.image_name}
            onUpload={onUpload}
            fontSize={36}
          />
        )}
        {imageDTO && (
          <>
            <DndImage imageDTO={imageDTO} borderRadius="base" borderWidth={1} borderStyle="solid" w="full" />
            <Flex position="absolute" flexDir="column" top={2} insetInlineEnd={2} gap={1}>
              <DndImageIcon
                onClick={handleResetControlImage}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
              />
            </Flex>
            <Flex position="absolute" flexDir="column" bottom={2} insetInlineEnd={2} gap={1}>
              <DndImageIcon
                onClick={recallSizeAndOptimize}
                icon={<PiRulerBold size={16} />}
                tooltip={t('parameters.useSize')}
                isDisabled={!imageDTO || (tab === 'canvas' && isStaging)}
              />
            </Flex>
          </>
        )}
        <DndDropTarget dndTarget={dndTarget} dndTargetData={dndTargetData} label={t('gallery.drop')} />
      </Flex>
    );
  }
);

RefImageImage.displayName = 'RefImageImage';
