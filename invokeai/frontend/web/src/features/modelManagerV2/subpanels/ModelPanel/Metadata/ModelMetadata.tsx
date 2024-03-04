import { Flex } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { useGetModelMetadataQuery } from 'services/api/endpoints/models';

export const ModelMetadata = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data: metadata } = useGetModelMetadataQuery(selectedModelKey ?? skipToken);

  return (
    <>
      <Flex flexDir="column" height="full" gap="3">
        <DataViewer label="metadata" data={metadata || {}} />
      </Flex>
    </>
  );
};
