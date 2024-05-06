import { Flex, IconButton, Spacer, Text, useShiftModifier } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { isFloatingImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useLayoutEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiHourglassBold, PiXBold } from 'react-icons/pi';
import { Rnd } from 'react-rnd';

const defaultDim = 256;
const maxDim = 512;
const defaultSize = { width: defaultDim, height: defaultDim + 24 };
const maxSize = { width: maxDim, height: maxDim + 24 };
const rndDefault = { x: 0, y: 0, ...defaultSize };

const rndStyles = {
  zIndex: 11,
};

const enableResizing = {
  top: false,
  right: false,
  bottom: false,
  left: false,
  topRight: false,
  bottomRight: true,
  bottomLeft: false,
  topLeft: false,
};

const FloatingImageViewerComponent = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const shift = useShiftModifier();
  const rndRef = useRef<Rnd>(null);
  const imagePreviewRef = useRef<HTMLDivElement>(null);
  const onClose = useCallback(() => {
    dispatch(isFloatingImageViewerOpenChanged(false));
  }, [dispatch]);

  const fitToScreen = useCallback(() => {
    if (!imagePreviewRef.current || !rndRef.current) {
      return;
    }
    const el = imagePreviewRef.current;
    const rnd = rndRef.current;

    const { top, right, bottom, left, width, height } = el.getBoundingClientRect();
    const { innerWidth, innerHeight } = window;

    const newPosition = rnd.getDraggablePosition();

    if (top < 0) {
      newPosition.y = 0;
    }
    if (left < 0) {
      newPosition.x = 0;
    }
    if (bottom > innerHeight) {
      newPosition.y = innerHeight - height;
    }
    if (right > innerWidth) {
      newPosition.x = innerWidth - width;
    }
    rnd.updatePosition(newPosition);
  }, []);

  const onDoubleClick = useCallback(() => {
    if (!rndRef.current || !imagePreviewRef.current) {
      return;
    }
    const { width, height } = imagePreviewRef.current.getBoundingClientRect();
    if (width === defaultSize.width && height === defaultSize.height) {
      rndRef.current.updateSize(maxSize);
    } else {
      rndRef.current.updateSize(defaultSize);
    }
    flushSync(fitToScreen);
  }, [fitToScreen]);

  useLayoutEffect(() => {
    window.addEventListener('resize', fitToScreen);
    return () => {
      window.removeEventListener('resize', fitToScreen);
    };
  }, [fitToScreen]);

  useLayoutEffect(() => {
    // Set the initial position
    if (!imagePreviewRef.current || !rndRef.current) {
      return;
    }

    const { width, height } = imagePreviewRef.current.getBoundingClientRect();

    const initialPosition = {
      // 54 = width of left-hand vertical bar of tab icons
      // 430 = width of parameters panel
      x: 54 + 430 / 2 - width / 2,
      // 16 = just a reasonable bottom padding
      y: window.innerHeight - height - 16,
    };

    rndRef.current.updatePosition(initialPosition);
  }, [fitToScreen]);

  return (
    <Rnd
      ref={rndRef}
      default={rndDefault}
      bounds="window"
      lockAspectRatio={shift}
      minWidth={defaultSize.width}
      minHeight={defaultSize.height}
      maxWidth={maxSize.width}
      maxHeight={maxSize.height}
      style={rndStyles}
      enableResizing={enableResizing}
    >
      <Flex
        ref={imagePreviewRef}
        flexDir="column"
        bg="base.850"
        borderRadius="base"
        w="full"
        h="full"
        borderWidth={1}
        shadow="dark-lg"
        cursor="move"
      >
        <Flex bg="base.800" w="full" p={1} onDoubleClick={onDoubleClick}>
          <Text fontSize="sm" fontWeight="semibold" color="base.300" ps={2}>
            {t('common.viewer')}
          </Text>
          <Spacer />
          <IconButton aria-label={t('common.close')} icon={<PiXBold />} size="sm" variant="link" onClick={onClose} />
        </Flex>
        <Flex p={2} w="full" h="full">
          <CurrentImagePreview
            isDragDisabled={true}
            isDropDisabled={true}
            withNextPrevButtons={false}
            withMetadata={false}
            alwaysShowProgress
          />
        </Flex>
      </Flex>
    </Rnd>
  );
});

FloatingImageViewerComponent.displayName = 'FloatingImageViewerComponent';

export const FloatingImageViewer = memo(() => {
  const isOpen = useAppSelector((s) => s.gallery.isFloatingImageViewerOpen);

  if (!isOpen) {
    return null;
  }

  return <FloatingImageViewerComponent />;
});

FloatingImageViewer.displayName = 'FloatingImageViewer';

export const ToggleFloatingImageViewerButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isOpen = useAppSelector((s) => s.gallery.isFloatingImageViewerOpen);

  const onToggle = useCallback(() => {
    dispatch(isFloatingImageViewerOpenChanged(!isOpen));
  }, [dispatch, isOpen]);

  return (
    <IconButton
      tooltip={isOpen ? t('gallery.closeFloatingViewer') : t('gallery.openFloatingViewer')}
      aria-label={isOpen ? t('gallery.closeFloatingViewer') : t('gallery.openFloatingViewer')}
      icon={<PiHourglassBold fontSize={16} />}
      size="sm"
      onClick={onToggle}
      variant="link"
      colorScheme={isOpen ? 'invokeBlue' : 'base'}
      boxSize={8}
    />
  );
});

ToggleFloatingImageViewerButton.displayName = 'ToggleFloatingImageViewerButton';
