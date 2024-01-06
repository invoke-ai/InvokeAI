import { ListItem, UnorderedList } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { some } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

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
      <InvText>{topMessage}</InvText>
      <UnorderedList paddingInlineStart={6}>
        {imageUsage.isInitialImage && (
          <ListItem>{t('common.img2img')}</ListItem>
        )}
        {imageUsage.isCanvasImage && (
          <ListItem>{t('common.unifiedCanvas')}</ListItem>
        )}
        {imageUsage.isControlImage && (
          <ListItem>{t('common.controlNet')}</ListItem>
        )}
        {imageUsage.isNodesImage && (
          <ListItem>{t('common.nodeEditor')}</ListItem>
        )}
      </UnorderedList>
      <InvText>{bottomMessage}</InvText>
    </>
  );
};

export default memo(ImageUsageMessage);
