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
      <UnorderedList paddingInlineStart={6} fontSize="sm">
        {imageUsage.isControlLayerImage && <ListItem>{t('controlLayers.controlLayer')}</ListItem>}
        {imageUsage.isReferenceImage && <ListItem>{t('controlLayers.referenceImage')}</ListItem>}
        {imageUsage.isInpaintMaskImage && <ListItem>{t('controlLayers.inpaintMask')}</ListItem>}
        {imageUsage.isRasterLayerImage && <ListItem>{t('controlLayers.rasterLayer')}</ListItem>}
        {imageUsage.isRegionalGuidanceImage && <ListItem>{t('controlLayers.regionalGuidance')}</ListItem>}
        {imageUsage.isUpscaleImage && <ListItem>{t('ui.tabs.upscalingTab')}</ListItem>}
        {imageUsage.isNodesImage && <ListItem>{t('ui.tabs.workflowsTab')}</ListItem>}
      </UnorderedList>
      <Text>{bottomMessage}</Text>
    </>
  );
};

export default memo(ImageUsageMessage);
