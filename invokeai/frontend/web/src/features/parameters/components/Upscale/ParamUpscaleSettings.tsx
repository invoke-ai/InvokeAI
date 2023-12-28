import { Flex, useDisclosure } from '@chakra-ui/react';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import {
  InvPopover,
  InvPopoverBody,
  InvPopoverContent,
  InvPopoverTrigger,
} from 'common/components/InvPopover/wrapper';
import { useIsAllowedToUpscale } from 'features/parameters/hooks/useIsAllowedToUpscale';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExpand } from 'react-icons/fa';
import type { ImageDTO } from 'services/api/types';

import ParamESRGANModel from './ParamRealESRGANModel';

type Props = { imageDTO?: ImageDTO };

const ParamUpscalePopover = (props: Props) => {
  const { imageDTO } = props;
  const dispatch = useAppDispatch();
  const inProgress = useIsQueueMutationInProgress();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { isAllowedToUpscale, detail } = useIsAllowedToUpscale(imageDTO);

  const handleClickUpscale = useCallback(() => {
    onClose();
    if (!imageDTO || !isAllowedToUpscale) {
      return;
    }
    dispatch(upscaleRequested({ imageDTO }));
  }, [dispatch, imageDTO, isAllowedToUpscale, onClose]);

  return (
    <InvPopover isOpen={isOpen} onClose={onClose}>
      <InvPopoverTrigger>
        <InvIconButton
          tooltip={t('parameters.upscale')}
          onClick={onOpen}
          icon={<FaExpand />}
          aria-label={t('parameters.upscale')}
        />
      </InvPopoverTrigger>
      <InvPopoverContent>
        <InvPopoverBody>
          <Flex
            sx={{
              flexDirection: 'column',
              gap: 4,
            }}
          >
            <ParamESRGANModel />
            <InvButton
              tooltip={detail}
              size="sm"
              isDisabled={!imageDTO || inProgress || !isAllowedToUpscale}
              onClick={handleClickUpscale}
            >
              {t('parameters.upscaleImage')}
            </InvButton>
          </Flex>
        </InvPopoverBody>
      </InvPopoverContent>
    </InvPopover>
  );
};

export default memo(ParamUpscalePopover);
