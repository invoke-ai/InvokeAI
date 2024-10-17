import { Alert, AlertDescription, AlertIcon, AlertTitle, Button, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useBoolean } from 'common/hooks/useBoolean';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useCurrentDestination } from 'features/queue/hooks/useCurrentDestination';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { activeTabCanvasRightPanelChanged, setActiveTab } from 'features/ui/store/uiSlice';
import { AnimatePresence, motion } from 'framer-motion';
import type { PropsWithChildren, ReactNode } from 'react';
import { useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';

const ActivateImageViewerButton = (props: PropsWithChildren) => {
  const imageViewer = useImageViewer();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    imageViewer.open();
    dispatch(activeTabCanvasRightPanelChanged('gallery'));
  }, [imageViewer, dispatch]);
  return (
    <Button onClick={onClick} size="sm" variant="link" color="base.50">
      {props.children}
    </Button>
  );
};

export const CanvasAlertsSendingToGallery = () => {
  const { t } = useTranslation();
  const destination = useCurrentDestination();
  const tab = useAppSelector(selectActiveTab);
  const isVisible = useMemo(() => {
    // This alert should only be visible when the destination is gallery and the tab is canvas
    if (tab !== 'canvas') {
      return false;
    }
    if (!destination) {
      return false;
    }

    return destination === 'gallery';
  }, [destination, tab]);

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
    dispatch(activeTabCanvasRightPanelChanged('layers'));
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
  const tab = useAppSelector(selectActiveTab);
  const isVisible = useMemo(() => {
    // When we are on a non-canvas tab, and the current generation's destination is not the canvas, we don't show the alert
    // For example, on the workflows tab, when the destinatin is gallery, we don't show the alert
    if (tab !== 'canvas' && destination !== 'canvas') {
      return false;
    }
    if (isStaging) {
      return true;
    }

    if (!destination) {
      return false;
    }

    return destination === 'canvas';
  }, [destination, isStaging, tab]);

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
  const isHovered = useBoolean(false);

  return (
    <AnimatePresence>
      {(isVisible || isHovered.isTrue) && (
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
            </Flex>
            <AlertDescription>{description}</AlertDescription>
          </Alert>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
