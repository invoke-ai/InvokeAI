import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Button,
  Flex,
  Icon,
  IconButton,
  Spacer,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useBoolean } from 'common/hooks/useBoolean';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  selectCanvasRightPanelGalleryTab,
  selectCanvasRightPanelLayersTab,
} from 'features/controlLayers/store/ephemeral';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useCurrentDestination } from 'features/queue/hooks/useCurrentDestination';
import { selectShowSendingToAlerts, showSendingToAlertsChanged } from 'features/system/store/systemSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { AnimatePresence, motion } from 'framer-motion';
import type { PropsWithChildren, ReactNode } from 'react';
import { useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const ActivateImageViewerButton = (props: PropsWithChildren) => {
  const imageViewer = useImageViewer();
  const onClick = useCallback(() => {
    imageViewer.open();
    selectCanvasRightPanelGalleryTab();
  }, [imageViewer]);
  return (
    <Button onClick={onClick} size="sm" variant="link" color="base.50">
      {props.children}
    </Button>
  );
};

export const CanvasAlertsSendingToGallery = () => {
  const { t } = useTranslation();
  const destination = useCurrentDestination();
  const isVisible = useMemo(() => {
    if (!destination) {
      return false;
    }

    if (destination === 'canvas') {
      return false;
    }

    return true;
  }, [destination]);

  return (
    <AlertWrapper
      title={t('controlLayers.sendingToGallery')}
      description={
        <Trans i18nKey="controlLayers.viewProgressInViewer" components={{ Btn: <ActivateImageViewerButton /> }} />
      }
      isVisible={isVisible}
    />
  );
};

const ActivateCanvasButton = (props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  const imageViewer = useImageViewer();
  const onClick = useCallback(() => {
    dispatch(setActiveTab('canvas'));
    selectCanvasRightPanelLayersTab();
    imageViewer.close();
  }, [dispatch, imageViewer]);
  return (
    <Button onClick={onClick} size="sm" variant="link" color="base.50">
      {props.children}
    </Button>
  );
};

export const CanvasAlertsSendingToCanvas = () => {
  const { t } = useTranslation();
  const destination = useCurrentDestination();
  const isStaging = useAppSelector(selectIsStaging);
  const isVisible = useMemo(() => {
    if (isStaging) {
      return true;
    }
    if (!destination) {
      return false;
    }

    if (destination !== 'canvas') {
      return false;
    }

    return true;
  }, [destination, isStaging]);

  return (
    <AlertWrapper
      title={t('controlLayers.sendingToCanvas')}
      description={
        <Trans i18nKey="controlLayers.viewProgressOnCanvas" components={{ Btn: <ActivateCanvasButton /> }} />
      }
      isVisible={isVisible}
    />
  );
};

const AlertWrapper = ({
  title,
  description,
  isVisible,
}: {
  title: ReactNode;
  description: ReactNode;
  isVisible: boolean;
}) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const showSendToAlerts = useAppSelector(selectShowSendingToAlerts);
  const isHovered = useBoolean(false);
  const onClickDontShowMeThese = useCallback(() => {
    dispatch(showSendingToAlertsChanged(false));
    isHovered.setFalse();
  }, [dispatch, isHovered]);

  return (
    <AnimatePresence>
      {(isVisible || isHovered.isTrue) && showSendToAlerts && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1, transition: { duration: 0.1, ease: 'easeOut' } }}
          exit={{
            opacity: 0,
            transition: { duration: 0.1, delay: !isHovered.isTrue ? 1 : 0.1, ease: 'easeIn' },
          }}
          onMouseEnter={isHovered.setTrue}
          onMouseLeave={isHovered.setFalse}
        >
          <Alert
            status="warning"
            flexDir="column"
            pointerEvents="auto"
            borderRadius="base"
            fontSize="sm"
            shadow="md"
            w="fit-content"
            alignSelf="flex-end"
          >
            <Flex w="full" alignItems="center">
              <AlertIcon />
              <AlertTitle>{title}</AlertTitle>
              <Spacer />
              <IconButton
                variant="link"
                icon={<Icon as={PiXBold} fill="base.50 !important" />}
                tooltip={t('common.dontShowMeThese')}
                aria-label={t('common.dontShowMeThese')}
                right={-1}
                top={-2}
                onClick={onClickDontShowMeThese}
                minW="auto"
              />
            </Flex>
            <AlertDescription>{description}</AlertDescription>
          </Alert>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
