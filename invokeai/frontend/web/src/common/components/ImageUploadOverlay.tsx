import { Box, Flex, Heading } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';

type ImageUploadOverlayProps = {
  isDragAccept: boolean;
  isDragReject: boolean;
  setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

const ImageUploadOverlay = (props: ImageUploadOverlayProps) => {
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
          opacity: 0.4,
          width: '100%',
          height: '100%',
          flexDirection: 'column',
          rowGap: 4,
          alignItems: 'center',
          justifyContent: 'center',
          bg: 'base.900',
          boxShadow: `inset 0 0 20rem 1rem var(--invokeai-colors-${
            isDragAccept ? 'accent' : 'error'
          }-500)`,
        }}
      >
        {isDragAccept ? (
          <Heading size="lg">Drop to Upload</Heading>
        ) : (
          <>
            <Heading size="lg">Invalid Upload</Heading>
            <Heading size="md">Must be single JPEG or PNG image</Heading>
          </>
        )}
      </Flex>
    </Box>
  );
};
export default ImageUploadOverlay;
