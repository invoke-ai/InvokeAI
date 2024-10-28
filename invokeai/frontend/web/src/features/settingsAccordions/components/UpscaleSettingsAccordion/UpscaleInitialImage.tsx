import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { Dnd } from 'features/dnd2/dnd';
import { DndDropTarget } from 'features/dnd2/DndDropTarget';
import { DndImage } from 'features/dnd2/DndImage';
import { DndImageIcon } from 'features/dnd2/DndImageIcon';
import { selectUpscaleInitialImage, upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import type { PostUploadAction } from 'services/api/types';

export const UpscaleInitialImage = () => {
  const dispatch = useAppDispatch();
  const imageDTO = useAppSelector(selectUpscaleInitialImage);
  const postUploadAction = useMemo<PostUploadAction>(
    () => ({
      type: 'SET_UPSCALE_INITIAL_IMAGE',
    }),
    []
  );
  const targetData = useMemo<Dnd.types['TargetDataTypeMap']['setUpscaleInitialImageFromImage']>(
    () => Dnd.Target.setUpscaleInitialImageFromImage.getData(),
    []
  );

  const onReset = useCallback(() => {
    dispatch(upscaleInitialImageChanged(null));
  }, [dispatch]);

  return (
    <Flex justifyContent="flex-start">
      <Flex position="relative" w={36} h={36} alignItems="center" justifyContent="center">
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
        <DndDropTarget targetData={targetData} label={t('gallery.drop')} />
      </Flex>
    </Flex>
  );
};
