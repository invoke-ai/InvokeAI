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
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { UpscaleWarning } from 'features/settingsAccordions/components/UpscaleSettingsAccordion/UpscaleWarning';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import ParamSpandrelModel from './ParamSpandrelModel';

type Props = { imageDTO?: ImageDTO };

const ParamUpscalePopover = (props: Props) => {
  const { imageDTO } = props;
  const dispatch = useAppDispatch();
  const { simpleUpscaleModel } = useAppSelector((s) => s.upscale);
  const inProgress = useIsQueueMutationInProgress();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleClickUpscale = useCallback(() => {
    onClose();
    if (!imageDTO) {
      return;
    }
    dispatch(upscaleRequested({ imageDTO }));
  }, [dispatch, imageDTO, onClose]);

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
        <PopoverBody w={96}>
          <Flex flexDirection="column" gap={4}>
            <ParamSpandrelModel isMultidiffusion={false} />
            <UpscaleWarning usesTile={false} />
            <Button
              size="sm"
              isDisabled={!imageDTO || inProgress || !simpleUpscaleModel}
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
