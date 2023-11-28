import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageWorkflowQuery } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import DataViewer from './DataViewer';

type Props = {
  image: ImageDTO;
};

const ImageMetadataWorkflowTabContent = ({ image }: Props) => {
  const { t } = useTranslation();
  const { currentData: workflow } = useGetImageWorkflowQuery(image.image_name);

  if (!workflow) {
    return <IAINoContentFallback label={t('nodes.noWorkflow')} />;
  }

  return <DataViewer data={workflow} label={t('metadata.workflow')} />;
};

export default memo(ImageMetadataWorkflowTabContent);
