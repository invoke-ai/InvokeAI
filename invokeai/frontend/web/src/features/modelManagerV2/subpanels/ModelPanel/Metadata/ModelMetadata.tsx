import { Box, Flex } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { useMemo } from 'react';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';
import type { ModelType } from 'services/api/types';

import { TriggerPhrases } from './TriggerPhrases';

const MODEL_TYPE_TRIGGER_PHRASE: ModelType[] = ['main', 'lora'];

export const ModelMetadata = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  const shouldShowTriggerPhraseSettings = useMemo(() => {
    if (!data?.type) {
      return false;
    }
    return MODEL_TYPE_TRIGGER_PHRASE.includes(data.type);
  }, [data]);

  return (
    <Flex flexDir="column" height="full" gap="3">
      {shouldShowTriggerPhraseSettings && (
        <Box layerStyle="second" borderRadius="base" p={3}>
          <TriggerPhrases />
        </Box>
      )}
      <DataViewer label="metadata" data={data?.source_api_response || {}} />
    </Flex>
  );
};
