import { Button, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import CurrentImageButtons from 'features/gallery/components/CurrentImage/CurrentImageButtons';
import CurrentImagePreview from 'features/gallery/components/CurrentImage/CurrentImagePreview';
import { isImageViewerOpenChanged } from 'features/gallery/store/gallerySlice';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowLeftBold } from 'react-icons/pi';

const TAB_NAME_TO_TKEY: Record<InvokeTabName, string> = {
  txt2img: 'common.txt2img',
  unifiedCanvas: 'common.unifiedCanvas',
  nodes: 'common.nodes',
  modelManager: 'modelManager.modelManager',
  queue: 'queue.queue',
};

export const ImageViewer = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isOpen = useAppSelector((s) => s.gallery.isImageViewerOpen);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const activeTabLabel = useMemo(
    () => t('gallery.backToEditor', { tab: t(TAB_NAME_TO_TKEY[activeTabName]) }),
    [t, activeTabName]
  );

  const onClose = useCallback(() => {
    dispatch(isImageViewerOpenChanged(false));
  }, [dispatch]);

  const onOpen = useCallback(() => {
    dispatch(isImageViewerOpenChanged(true));
  }, [dispatch]);

  useHotkeys('esc', onClose, { enabled: isOpen }, [isOpen]);
  useHotkeys('i', onOpen, { enabled: !isOpen }, [isOpen]);

  if (!isOpen) {
    return null;
  }

  return (
    <Flex
      layerStyle="first"
      borderRadius="base"
      position="absolute"
      flexDirection="column"
      top={0}
      right={0}
      bottom={0}
      left={0}
      p={2}
      rowGap={4}
      alignItems="center"
      justifyContent="center"
      zIndex={10} // reactflow puts its minimap at 5, so we need to be above that
    >
      <CurrentImageButtons />
      <CurrentImagePreview />
      <Button
        aria-label={activeTabLabel}
        tooltip={activeTabLabel}
        onClick={onClose}
        leftIcon={<PiArrowLeftBold />}
        position="absolute"
        top={2}
        insetInlineEnd={2}
      >
        {t('common.back')}
      </Button>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
