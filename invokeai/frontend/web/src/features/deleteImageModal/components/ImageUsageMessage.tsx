import { ListItem, Text, UnorderedList } from '@chakra-ui/react';
import { some } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { ImageUsage } from '../store/types';

type Props = {
  imageUsage?: ImageUsage;
  topMessage?: string;
  bottomMessage?: string;
};
const ImageUsageMessage = (props: Props) => {
  const { t } = useTranslation();
  const {
    imageUsage,
    topMessage = t('gallery.currentlyInUse'),
    bottomMessage = t('gallery.featuresWillReset'),
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
        {imageUsage.isInitialImage && (
          <ListItem>{t('common.img2img')}</ListItem>
        )}
        {imageUsage.isCanvasImage && (
          <ListItem>{t('common.unifiedCanvas')}</ListItem>
        )}
        {imageUsage.isControlNetImage && (
          <ListItem>{t('common.controlNet')}</ListItem>
        )}
        {imageUsage.isIPAdapterImage && (
          <ListItem>{t('common.ipAdapter')}</ListItem>
        )}
        {imageUsage.isNodesImage && (
          <ListItem>{t('common.nodeEditor')}</ListItem>
        )}
      </UnorderedList>
      <Text>{bottomMessage}</Text>
    </>
  );
};

export default memo(ImageUsageMessage);
