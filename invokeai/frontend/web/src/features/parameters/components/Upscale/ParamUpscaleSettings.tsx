import {
  Button,
  Flex,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch } from 'app/store/storeHooks';
import { useIsAllowedToUpscale } from 'features/parameters/hooks/useIsAllowedToUpscale';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';
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
    <Popover isOpen={isOpen} onClose={onClose} isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={t('parameters.upscale')}
          onClick={onOpen}
          icon={<PiFrameCornersBold />}
          aria-label={t('parameters.upscale')}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minW={96}>
          <Flex flexDirection="column" gap={4}>
            <ParamESRGANModel />
            <Button
              tooltip={detail}
              size="sm"
              isDisabled={!imageDTO || inProgress || !isAllowedToUpscale}
              onClick={handleClickUpscale}
            >
              {t('parameters.upscaleImage')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ParamUpscalePopover);
