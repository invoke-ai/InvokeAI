import { Flex, useDisclosure } from '@chakra-ui/react';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExpandArrowsAlt } from 'react-icons/fa';
import { ImageDTO } from 'services/api/types';
import ParamESRGANModel from './ParamRealESRGANModel';

type Props = { imageDTO?: ImageDTO };

const ParamUpscalePopover = (props: Props) => {
  const { imageDTO } = props;
  const dispatch = useAppDispatch();
  const isBusy = useAppSelector(selectIsBusy);
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleClickUpscale = useCallback(() => {
    onClose();
    if (!imageDTO) {
      return;
    }
    dispatch(upscaleRequested({ image_name: imageDTO.image_name }));
  }, [dispatch, imageDTO, onClose]);

  return (
    <IAIPopover
      isOpen={isOpen}
      onClose={onClose}
      triggerComponent={
        <IAIIconButton
          onClick={onOpen}
          icon={<FaExpandArrowsAlt />}
          aria-label={t('parameters.upscale')}
        />
      }
    >
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 4,
        }}
      >
        <ParamESRGANModel />
        <IAIButton
          size="sm"
          isDisabled={!imageDTO || isBusy}
          onClick={handleClickUpscale}
        >
          {t('parameters.upscaleImage')}
        </IAIButton>
      </Flex>
    </IAIPopover>
  );
};

export default memo(ParamUpscalePopover);
