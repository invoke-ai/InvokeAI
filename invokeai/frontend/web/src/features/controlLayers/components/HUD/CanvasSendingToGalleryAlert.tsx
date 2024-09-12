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
import {
  setRightPanelTabToGallery,
  setRightPanelTabToLayers,
} from 'features/controlLayers/components/CanvasRightPanel';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useCurrentDestination } from 'features/queue/hooks/useCurrentDestination';
import { selectShowSendToAlerts, showSendToAlertsChanged } from 'features/system/store/systemSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { PropsWithChildren } from 'react';
import { useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const DontShowMeTheseAgainButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(showSendToAlertsChanged(false));
  }, [dispatch]);
  return (
    <IconButton
      variant="link"
      icon={<Icon as={PiXBold} fill="base.50 !important" />}
      tooltip={t('common.dontShowMeThese')}
      aria-label={t('common.dontShowMeThese')}
      right={-1}
      top={-2}
      onClick={onClick}
      minW="auto"
    />
  );
};

const ActivateImageViewerButton = (props: PropsWithChildren) => {
  const imageViewer = useImageViewer();
  const onClick = useCallback(() => {
    imageViewer.open();
    setRightPanelTabToGallery();
  }, [imageViewer]);
  return (
    <Button onClick={onClick} size="sm" variant="link" color="base.50">
      {props.children}
    </Button>
  );
};

export const SendingToGalleryAlert = () => {
  const { t } = useTranslation();
  const destination = useCurrentDestination();
  const showSendToAlerts = useAppSelector(selectShowSendToAlerts);

  if (!showSendToAlerts) {
    return null;
  }

  if (!destination) {
    return null;
  }

  if (destination === 'canvas') {
    return null;
  }

  return (
    <Alert status="warning" flexDir="column" pointerEvents="auto" borderRadius="base" fontSize="sm" shadow="md">
      <Flex w="full" alignItems="center">
        <AlertIcon />
        <AlertTitle>{t('controlLayers.sendingToGallery')}</AlertTitle>
        <Spacer />
        <DontShowMeTheseAgainButton />
      </Flex>
      <AlertDescription>
        <Trans i18nKey="controlLayers.viewProgressInViewer" components={{ Btn: <ActivateImageViewerButton /> }} />
      </AlertDescription>
    </Alert>
  );
};

const ActivateCanvasButton = (props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  const imageViewer = useImageViewer();
  const onClick = useCallback(() => {
    dispatch(setActiveTab('generation'));
    setRightPanelTabToLayers();
    imageViewer.close();
  }, [dispatch, imageViewer]);
  return (
    <Button onClick={onClick} size="sm" variant="link" color="base.50">
      {props.children}
    </Button>
  );
};

export const SendingToCanvasAlert = () => {
  const { t } = useTranslation();
  const destination = useCurrentDestination();
  const showSendToAlerts = useAppSelector(selectShowSendToAlerts);

  if (!showSendToAlerts) {
    return null;
  }

  if (!destination) {
    return null;
  }

  if (destination !== 'canvas') {
    return null;
  }

  return (
    <Alert status="warning" flexDir="column" pointerEvents="auto" borderRadius="base" fontSize="sm" shadow="md">
      <Flex w="full" alignItems="center">
        <AlertIcon />
        <AlertTitle>{t('controlLayers.sendingToCanvas')}</AlertTitle>
        <Spacer />
        <DontShowMeTheseAgainButton />
      </Flex>
      <AlertDescription>
        <Trans i18nKey="controlLayers.viewProgressOnCanvas" components={{ Btn: <ActivateCanvasButton /> }} />
      </AlertDescription>
    </Alert>
  );
};
