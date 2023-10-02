import { Flex, useDisclosure } from '@chakra-ui/react';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { useMayUpscale } from 'features/parameters/hooks/useMayUpscale';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExpand } from 'react-icons/fa';
import { ImageDTO } from 'services/api/types';
import ParamESRGANModel from './ParamRealESRGANModel';

type Props = { imageDTO?: ImageDTO };

const ParamUpscalePopover = (props: Props) => {
  const { imageDTO } = props;
  const dispatch = useAppDispatch();
  const inProgress = useIsQueueMutationInProgress();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { mayUpscale, detail } = useMayUpscale(imageDTO);

  const handleClickUpscale = useCallback(() => {
    onClose();
    if (!imageDTO || !mayUpscale) {
      return;
    }
    dispatch(upscaleRequested({ imageDTO }));
  }, [dispatch, imageDTO, mayUpscale, onClose]);

  return (
    <IAIPopover
      isOpen={isOpen}
      onClose={onClose}
      triggerComponent={
        <IAIIconButton
          tooltip={t('parameters.upscale')}
          onClick={onOpen}
          icon={<FaExpand />}
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
          tooltip={detail}
          size="sm"
          isDisabled={!imageDTO || inProgress || !mayUpscale}
          onClick={handleClickUpscale}
        >
          {t('parameters.upscaleImage')}
        </IAIButton>
      </Flex>
    </IAIPopover>
  );
};

export default memo(ParamUpscalePopover);
