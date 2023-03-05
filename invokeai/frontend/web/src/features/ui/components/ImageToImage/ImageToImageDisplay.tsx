import { ChakraProps, Flex, Grid } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import ImageUploadButton from 'common/components/ImageUploaderButton';
import CurrentImageDisplay from 'features/gallery/components/CurrentImageDisplay';
import InitImagePreview from './InitImagePreview';

const workareaSplitViewStyle: ChakraProps['sx'] = {
  flexDirection: 'column',
  height: '100%',
  width: '100%',
  gap: 4,

  padding: 4,
};

const ImageToImageDisplay = () => {
  const initialImage = useAppSelector(
    (state: RootState) => state.generation.initialImage
  );

  const imageToImageComponent = initialImage ? (
    <Flex flexDirection="column" gap={4} w="100%" h="100%">
      <InitImagePreview />
    </Flex>
  ) : (
    <ImageUploadButton />
  );

  return (
    <Grid
      sx={{
        w: '100%',
        h: '100%',
        gridTemplateColumns: '1fr 1fr',
        borderRadius: 'base',
        bg: 'base.850',
      }}
    >
      <Flex sx={{ ...workareaSplitViewStyle, paddingInlineEnd: 2 }}>
        {imageToImageComponent}
      </Flex>
      <Flex sx={{ ...workareaSplitViewStyle, paddingInlineStart: 2 }}>
        <CurrentImageDisplay />
      </Flex>
    </Grid>
  );
};

export default ImageToImageDisplay;
