import { Flex } from '@chakra-ui/layout';
import { Portal } from '@chakra-ui/portal';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { RefObject } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { MdPhotoLibrary } from 'react-icons/md';
import type { ImperativePanelHandle } from 'react-resizable-panels';

type Props = {
  isGalleryCollapsed: boolean;
  galleryPanelRef: RefObject<ImperativePanelHandle>;
};

const FloatingGalleryButton = ({
  isGalleryCollapsed,
  galleryPanelRef,
}: Props) => {
  const { t } = useTranslation();

  const handleShowGallery = useCallback(() => {
    galleryPanelRef.current?.expand();
  }, [galleryPanelRef]);

  if (!isGalleryCollapsed) {
    return null;
  }

  return (
    <Portal>
      <Flex
        pos="absolute"
        transform="translate(0, -50%)"
        minW={8}
        top="50%"
        insetInlineEnd="1.63rem"
      >
        <InvIconButton
          tooltip="Show Gallery (G)"
          aria-label={t('accessibility.showGalleryPanel')}
          onClick={handleShowGallery}
          icon={<MdPhotoLibrary />}
          p={0}
          px={3}
          h={48}
          borderEndRadius={0}
        />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingGalleryButton);
