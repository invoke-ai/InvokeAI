import { Box, Flex, Heading } from '@chakra-ui/react';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

type ImageUploadOverlayProps = {
  isDragAccept: boolean;
  isDragReject: boolean;
  setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

const ImageUploadOverlay = (props: ImageUploadOverlayProps) => {
  const { t } = useTranslation();
  const {
    isDragAccept,
    isDragReject: _isDragAccept,
    setIsHandlingUpload,
  } = props;

  useHotkeys('esc', () => {
    setIsHandlingUpload(false);
  });

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        insetInlineStart: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 999,
        backdropFilter: 'blur(20px)',
      }}
    >
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          w: 'full',
          h: 'full',
          bg: 'base.900',
          opacity: 0.7,
          alignItems: 'center',
          justifyContent: 'center',
          transitionProperty: 'common',
          transitionDuration: '0.1s',
        }}
      />
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          width: 'full',
          height: 'full',
          alignItems: 'center',
          justifyContent: 'center',
          p: 4,
        }}
      >
        <Flex
          sx={{
            width: 'full',
            height: 'full',
            alignItems: 'center',
            justifyContent: 'center',
            flexDir: 'column',
            gap: 4,
            borderWidth: 3,
            borderRadius: 'xl',
            borderStyle: 'dashed',
            color: 'base.100',
            borderColor: 'base.200',
          }}
        >
          {isDragAccept ? (
            <Heading size="lg">{t('gallery.dropToUpload')}</Heading>
          ) : (
            <>
              <Heading size="lg">{t('toast.invalidUpload')}</Heading>
              <Heading size="md">
                {t('toast.uploadFailedInvalidUploadDesc')}
              </Heading>
            </>
          )}
        </Flex>
      </Flex>
    </Box>
  );
};
export default memo(ImageUploadOverlay);
