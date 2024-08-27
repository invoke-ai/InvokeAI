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
import { adHocPostProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $installModelsTab } from 'features/modelManagerV2/subpanels/InstallModels';
import ParamPostProcessingModel from 'features/parameters/components/PostProcessing/ParamPostProcessingModel';
import { selectPostProcessingModel } from 'features/parameters/store/upscaleSlice';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiFrameCornersBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = { imageDTO?: ImageDTO };

export const PostProcessingPopover = memo((props: Props) => {
  const { imageDTO } = props;
  const dispatch = useAppDispatch();
  const postProcessingModel = useAppSelector(selectPostProcessingModel);
  const inProgress = useIsQueueMutationInProgress();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleClickUpscale = useCallback(() => {
    onClose();
    if (!imageDTO) {
      return;
    }
    dispatch(adHocPostProcessingRequested({ imageDTO }));
  }, [dispatch, imageDTO, onClose]);

  return (
    <Popover isOpen={isOpen} onClose={onClose} isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={t('parameters.postProcessing')}
          onClick={onOpen}
          icon={<PiFrameCornersBold />}
          aria-label={t('parameters.postProcessing')}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody w={96}>
          <Flex flexDirection="column" gap={4}>
            <ParamPostProcessingModel />
            {!postProcessingModel && <MissingModelWarning />}
            <Button size="sm" isDisabled={!imageDTO || inProgress || !postProcessingModel} onClick={handleClickUpscale}>
              {t('parameters.processImage')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

PostProcessingPopover.displayName = 'PostProcessingPopover';

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
          i18nKey="upscaling.postProcessingMissingModelWarning"
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
