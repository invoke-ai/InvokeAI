import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { objectEquals } from '@observ33r/object-equals';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { UploadImageIconButton } from 'common/hooks/useImageUploadButton';
import { bboxSizeOptimized, bboxSizeRecalled } from 'features/controlLayers/store/canvasSlice';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { sizeOptimized, sizeRecalled } from 'features/controlLayers/store/paramsSlice';
import type { CroppableImageWithDims } from 'features/controlLayers/store/types';
import { imageDTOToCroppableImage, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { Editor } from 'features/cropper/lib/editor';
import { cropImageModalApi } from 'features/cropper/store';
import type { setGlobalReferenceImageDndTarget, setRegionalGuidanceReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiCropBold, PiRulerBold } from 'react-icons/pi';
import { useGetImageDTOQuery, useUploadImageMutation } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

type Props<T extends typeof setGlobalReferenceImageDndTarget | typeof setRegionalGuidanceReferenceImageDndTarget> = {
  image: CroppableImageWithDims | null;
  onChangeImage: (croppableImage: CroppableImageWithDims | null) => void;
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
    const imageWithDims = image?.crop?.image ?? image?.original.image ?? null;
    const croppedImageDTOReq = useGetImageDTOQuery(image?.crop?.image?.image_name ?? skipToken);
    const originalImageDTOReq = useGetImageDTOQuery(image?.original.image.image_name ?? skipToken);
    const [uploadImage] = useUploadImageMutation();

    const originalImageDTO = originalImageDTOReq.currentData;
    const croppedImageDTO = croppedImageDTOReq.currentData;
    const imageDTO = croppedImageDTO ?? originalImageDTO;

    const handleResetControlImage = useCallback(() => {
      onChangeImage(null);
    }, [onChangeImage]);

    useEffect(() => {
      if ((isConnected && croppedImageDTOReq.isError) || originalImageDTOReq.isError) {
        handleResetControlImage();
      }
    }, [handleResetControlImage, isConnected, croppedImageDTOReq.isError, originalImageDTOReq.isError]);

    const onUpload = useCallback(
      (imageDTO: ImageDTO) => {
        onChangeImage(imageDTOToCroppableImage(imageDTO));
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

    const edit = useCallback(() => {
      if (!originalImageDTO) {
        return;
      }

      // We will create a new editor instance each time the user wants to edit
      const editor = new Editor();

      // When the user applies the crop, we will upload the cropped image and store the applied crop box so if the user
      // re-opens the editor they see the same crop
      const onApplyCrop = async () => {
        const box = editor.getCropBox();
        if (objectEquals(box, image?.crop?.box)) {
          // If the box hasn't changed, don't do anything
          return;
        }
        if (!box || objectEquals(box, { x: 0, y: 0, width: originalImageDTO.width, height: originalImageDTO.height })) {
          // There is a crop applied but it is the whole iamge - revert to original image
          onChangeImage(imageDTOToCroppableImage(originalImageDTO));
          return;
        }
        const blob = await editor.exportImage('blob');
        const file = new File([blob], 'image.png', { type: 'image/png' });

        const newCroppedImageDTO = await uploadImage({
          file,
          is_intermediate: true,
          image_category: 'user',
        }).unwrap();

        onChangeImage(
          imageDTOToCroppableImage(originalImageDTO, {
            image: imageDTOToImageWithDims(newCroppedImageDTO),
            box,
            ratio: editor.getCropAspectRatio(),
          })
        );
      };

      const onReady = async () => {
        const initial = image?.crop ? { cropBox: image.crop.box, aspectRatio: image.crop.ratio } : undefined;
        // Load the image into the editor and open the modal once it's ready
        await editor.loadImage(originalImageDTO.image_url, initial);
      };

      cropImageModalApi.open({ editor, onApplyCrop, onReady });
    }, [image?.crop, onChangeImage, originalImageDTO, uploadImage]);

    return (
      <Flex
        position="relative"
        w="full"
        h="full"
        alignItems="center"
        data-error={!imageDTO && !imageWithDims?.image_name}
      >
        {!imageDTO && (
          <UploadImageIconButton
            w="full"
            h="full"
            isError={!imageDTO && !imageWithDims?.image_name}
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

            <Flex position="absolute" flexDir="column" top={2} insetInlineStart={2} gap={1}>
              <DndImageIcon
                onClick={edit}
                icon={<PiCropBold size={16} />}
                tooltip={t('common.crop')}
                isDisabled={!imageDTO}
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
