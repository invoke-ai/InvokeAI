import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { UploadImageButton } from 'common/hooks/useImageUploadButton';
import type { SetUpscaleInitialImageDndTargetData } from 'features/dnd/dnd';
import { setUpscaleInitialImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { selectUpscaleInitialImage, upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

export const UpscaleInitialImage = () => {
  const dispatch = useAppDispatch();
  const imageDTO = useAppSelector(selectUpscaleInitialImage);
  const dndTargetData = useMemo<SetUpscaleInitialImageDndTargetData>(
    () => setUpscaleInitialImageDndTarget.getData(),
    []
  );

  const onReset = useCallback(() => {
    dispatch(upscaleInitialImageChanged(null));
  }, [dispatch]);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(upscaleInitialImageChanged(imageDTO));
    },
    [dispatch]
  );

  return (
    <Flex justifyContent="flex-start">
      <Flex position="relative" w={36} h={36} alignItems="center" justifyContent="center">
        {!imageDTO && <UploadImageButton w="full" h="full" isError={!imageDTO} onUpload={onUpload} fontSize={36} />}
        {imageDTO && (
          <>
            <DndImage imageDTO={imageDTO} />
            <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
              <DndImageIcon
                onClick={onReset}
                icon={<PiArrowCounterClockwiseBold size={16} />}
                tooltip={t('common.reset')}
              />
            </Flex>
            <Text
              position="absolute"
              background="base.900"
              color="base.50"
              fontSize="sm"
              fontWeight="semibold"
              bottom={0}
              left={0}
              opacity={0.7}
              px={2}
              lineHeight={1.25}
              borderTopEndRadius="base"
              borderBottomStartRadius="base"
              pointerEvents="none"
            >{`${imageDTO.width}x${imageDTO.height}`}</Text>
          </>
        )}
        <DndDropTarget
          dndTarget={setUpscaleInitialImageDndTarget}
          dndTargetData={dndTargetData}
          label={t('gallery.drop')}
        />
      </Flex>
    </Flex>
  );
};
