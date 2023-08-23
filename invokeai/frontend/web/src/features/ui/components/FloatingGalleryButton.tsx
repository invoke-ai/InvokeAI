import { Flex } from '@chakra-ui/layout';
import { Portal } from '@chakra-ui/portal';
import IAIIconButton from 'common/components/IAIIconButton';
import { RefObject, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdPhotoLibrary } from 'react-icons/md';
import { ImperativePanelHandle } from 'react-resizable-panels';

type Props = {
  isGalleryCollapsed: boolean;
  galleryPanelRef: RefObject<ImperativePanelHandle>;
};

const FloatingGalleryButton = ({
  isGalleryCollapsed,
  galleryPanelRef,
}: Props) => {
  const { t } = useTranslation();

  const handleShowGallery = () => {
    galleryPanelRef.current?.expand();
  };

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
        <IAIIconButton
          tooltip="Show Gallery (G)"
          tooltipProps={{ placement: 'top' }}
          aria-label={t('common.showGalleryPanel')}
          onClick={handleShowGallery}
          icon={<MdPhotoLibrary />}
          sx={{
            p: 0,
            px: 3,
            h: 48,
            borderStartEndRadius: 0,
            borderEndEndRadius: 0,
            shadow: '2xl',
          }}
        />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingGalleryButton);
