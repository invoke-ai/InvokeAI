import { ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
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
      <Text>{topMessage}</Text>
      <UnorderedList paddingInlineStart={6}>
        {imageUsage.isLayerImage && <ListItem>{t('controlLayers.layers')}</ListItem>}
        {imageUsage.isControlAdapterImage && <ListItem>{t('controlLayers.controlAdapters')}</ListItem>}
        {imageUsage.isIPAdapterImage && <ListItem>{t('controlLayers.ipAdapters')}</ListItem>}
        {imageUsage.isNodesImage && <ListItem>{t('ui.tabs.workflowsTab')}</ListItem>}
      </UnorderedList>
      <Text>{bottomMessage}</Text>
    </>
  );
};

export default memo(ImageUsageMessage);
