import { ListItem, Text, UnorderedList } from '@chakra-ui/react';
import { some } from 'lodash-es';
import { memo } from 'react';
import { ImageUsage } from '../store/types';
type Props = {
  imageUsage?: ImageUsage;
  topMessage?: string;
  bottomMessage?: string;
};
const ImageUsageMessage = (props: Props) => {
  const {
    imageUsage,
    topMessage = 'This image is currently in use in the following features:',
    bottomMessage = 'If you delete this image, those features will immediately be reset.',
  } = props;

  if (!imageUsage) {
    return null;
  }

  if (!some(imageUsage)) {
    return null;
  }

  return (
    <>
      <Text>{topMessage}</Text>
      <UnorderedList sx={{ paddingInlineStart: 6 }}>
        {imageUsage.isInitialImage && <ListItem>Image to Image</ListItem>}
        {imageUsage.isCanvasImage && <ListItem>Unified Canvas</ListItem>}
        {imageUsage.isControlNetImage && <ListItem>ControlNet</ListItem>}
        {imageUsage.isNodesImage && <ListItem>Node Editor</ListItem>}
      </UnorderedList>
      <Text>{bottomMessage}</Text>
    </>
  );
};

export default memo(ImageUsageMessage);
