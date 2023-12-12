import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedImageWorkflow } from 'services/api/hooks/useDebouncedImageWorkflow';
import { ImageDTO } from 'services/api/types';
import DataViewer from './DataViewer';

type Props = {
  image: ImageDTO;
};

const ImageMetadataWorkflowTabContent = ({ image }: Props) => {
  const { t } = useTranslation();
  const { workflow } = useDebouncedImageWorkflow(image);

  if (!workflow) {
    return <IAINoContentFallback label={t('nodes.noWorkflow')} />;
  }

  return <DataViewer data={workflow} label={t('metadata.workflow')} />;
};

export default memo(ImageMetadataWorkflowTabContent);
