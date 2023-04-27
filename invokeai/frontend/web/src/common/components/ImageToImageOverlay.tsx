import { Badge, Box, ButtonGroup, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaUndo, FaUpload } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';
import { Image } from 'app/invokeai';

type ImageToImageOverlayProps = {
  setIsLoaded: (isLoaded: boolean) => void;
  image: Image;
};

const ImageToImageOverlay = ({
  setIsLoaded,
  image,
}: ImageToImageOverlayProps) => {
  const isImageToImageEnabled = useAppSelector(
    (state: RootState) => state.generation.isImageToImageEnabled
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleResetInitialImage = useCallback(() => {
    dispatch(clearInitialImage());
    setIsLoaded(false);
  }, [dispatch, setIsLoaded]);

  return (
    <Box
      sx={{
        top: 0,
        left: 0,
        w: 'full',
        h: 'full',
        position: 'absolute',
      }}
    >
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          p: 2,
          alignItems: 'flex-start',
        }}
      >
        <Badge variant="solid" colorScheme="base">
          {image.metadata?.width} × {image.metadata?.height}
        </Badge>
      </Flex>
    </Box>
  );
};

export default ImageToImageOverlay;
