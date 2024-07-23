import {
  Button,
  Flex,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $installModelsTab } from 'features/modelManagerV2/subpanels/InstallModels';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';
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
            {!simpleUpscaleModel && <MissingModelWarning />}
            <Button size="sm" isDisabled={!imageDTO || inProgress || !simpleUpscaleModel} onClick={handleClickUpscale}>
              {t('parameters.upscaleImage')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ParamUpscalePopover);

const MissingModelWarning = () => {
  const dispatch = useAppDispatch();

  const handleGoToModelManager = useCallback(() => {
    dispatch(setActiveTab('models'));
    $installModelsTab.set(3);
  }, [dispatch]);

  return (
    <Flex bg="error.500" borderRadius="base" padding={4} direction="column" fontSize="sm" gap={2}>
      <Text>
        <Trans
          i18nKey="upscaling.simpleUpscaleMissingModelWarning"
          components={{
            LinkComponent: (
              <Button size="sm" flexGrow={0} variant="link" color="base.50" onClick={handleGoToModelManager} />
            ),
          }}
        />
      </Text>
    </Flex>
  );
};
