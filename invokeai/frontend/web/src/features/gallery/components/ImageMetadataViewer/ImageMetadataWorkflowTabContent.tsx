import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedImageWorkflow } from 'services/api/hooks/useDebouncedImageWorkflow';
import type { ImageDTO } from 'services/api/types';

import DataViewer from './DataViewer';

type Props = {
  image: ImageDTO;
};

const ImageMetadataWorkflowTabContent = ({ image }: Props) => {
  const { t } = useTranslation();
  const { currentData } = useDebouncedImageWorkflow(image);
  const workflow = useMemo(() => {
    if (currentData?.workflow) {
      try {
        return JSON.parse(currentData.workflow);
      } catch {
        return null;
      }
    }
    return null;
  }, [currentData]);

  if (!workflow) {
    return <IAINoContentFallback label={t('nodes.noWorkflow')} />;
  }

  return (
    <DataViewer
      fileName={`${image.image_name.replace('.png', '')}_workflow`}
      data={workflow}
      label={t('metadata.workflow')}
    />
  );
};

export default memo(ImageMetadataWorkflowTabContent);
