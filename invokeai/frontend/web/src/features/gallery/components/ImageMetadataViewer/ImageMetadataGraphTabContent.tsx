import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedImageWorkflow } from 'services/api/hooks/useDebouncedImageWorkflow';
import type { ImageDTO } from 'services/api/types';

import DataViewer from './DataViewer';

type Props = {
  image: ImageDTO;
};

const ImageMetadataGraphTabContent = ({ image }: Props) => {
  const { t } = useTranslation();
  const { currentData } = useDebouncedImageWorkflow(image);
  const graph = useMemo(() => {
    if (currentData?.graph) {
      try {
        return JSON.parse(currentData.graph);
      } catch {
        return null;
      }
    }
    return null;
  }, [currentData]);

  if (!graph) {
    return <IAINoContentFallback label={t('nodes.noGraph')} />;
  }

  return (
    <DataViewer fileName={`${image.image_name.replace('.png', '')}_graph`} data={graph} label={t('nodes.graph')} />
  );
};

export default memo(ImageMetadataGraphTabContent);
