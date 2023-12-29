import { Flex } from '@chakra-ui/layout';
import { Portal } from '@chakra-ui/portal';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdPhotoLibrary } from 'react-icons/md';

type Props = {
  isGalleryCollapsed: boolean;
  expandGallery: () => void;
};

const FloatingGalleryButton = ({
  isGalleryCollapsed,
  expandGallery,
}: Props) => {
  const { t } = useTranslation();

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
          onClick={expandGallery}
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
