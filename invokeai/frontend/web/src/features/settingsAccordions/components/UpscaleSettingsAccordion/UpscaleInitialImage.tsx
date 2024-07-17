import { Flex } from '@invoke-ai/ui-library';
import { useCallback, useMemo } from 'react';
import IAIDndImage from '../../../../common/components/IAIDndImage';
import { PostUploadAction } from '../../../../services/api/types';
import { TypesafeDroppableData } from '../../../dnd/types';
import { useAppDispatch, useAppSelector } from '../../../../app/store/storeHooks';
import { t } from 'i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import IAIDndImageIcon from '../../../../common/components/IAIDndImageIcon';
import { upscaleInitialImageChanged } from '../../../parameters/store/upscaleSlice';

export const UpscaleInitialImage = () => {
  const dispatch = useAppDispatch();
  const imageDTO = useAppSelector((s) => s.upscale.upscaleInitialImage);

  const droppableData = useMemo<TypesafeDroppableData | undefined>(
    () => ({
      actionType: 'SET_UPSCALE_INITIAL_IMAGE',
      id: 'upscale-intial-image',
    }),
    []
  );

  const postUploadAction = useMemo<PostUploadAction>(
    () => ({
      type: 'SET_UPSCALE_INITIAL_IMAGE',
    }),
    []
  );

  const onReset = useCallback(() => {
    dispatch(upscaleInitialImageChanged(null));
  }, [dispatch]);

  return (
    <Flex justifyContent="flex-start">
      <Flex position="relative" w={36} h={36} alignItems="center" justifyContent="center">
        <IAIDndImage
          droppableData={droppableData}
          imageDTO={imageDTO || undefined}
          postUploadAction={postUploadAction}
        />
        {imageDTO && (
          <Flex position="absolute" flexDir="column" top={1} insetInlineEnd={1} gap={1}>
            <IAIDndImageIcon
              onClick={onReset}
              icon={<PiArrowCounterClockwiseBold size={16} />}
              tooltip={t('controlnet.resetControlImage')}
            />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
};
