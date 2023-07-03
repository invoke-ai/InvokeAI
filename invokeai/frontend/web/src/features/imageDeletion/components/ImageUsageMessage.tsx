import { some } from 'lodash-es';
import { memo } from 'react';
import { ImageUsage } from '../store/imageDeletionSlice';
import { ListItem, Text, UnorderedList } from '@chakra-ui/react';

const ImageUsageMessage = (props: { imageUsage?: ImageUsage }) => {
  const { imageUsage } = props;

  if (!imageUsage) {
    return null;
  }

  if (!some(imageUsage)) {
    return null;
  }

  return (
    <>
      <Text>This image is currently in use in the following features:</Text>
      <UnorderedList sx={{ paddingInlineStart: 6 }}>
        {imageUsage.isInitialImage && <ListItem>Image to Image</ListItem>}
        {imageUsage.isCanvasImage && <ListItem>Unified Canvas</ListItem>}
        {imageUsage.isControlNetImage && <ListItem>ControlNet</ListItem>}
        {imageUsage.isNodesImage && <ListItem>Node Editor</ListItem>}
      </UnorderedList>
      <Text>
        If you delete this image, those features will immediately be reset.
      </Text>
    </>
  );
};

export default memo(ImageUsageMessage);
