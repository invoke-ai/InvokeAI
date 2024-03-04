import { Box, Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

import { ModelMetadata } from './Metadata/ModelMetadata';
import { ModelAttrView } from './ModelAttrView';
import { ModelEdit } from './ModelEdit';
import { ModelView } from './ModelView';

export const Model = () => {
  const { t } = useTranslation();
  const selectedModelMode = useAppSelector((s) => s.modelmanagerV2.selectedModelMode);
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }

  return (
    <>
      <Flex flexDir="column" gap={1} p={2}>
        <Heading as="h2" fontSize="lg">
          {data.name}
        </Heading>

        {data.source && (
          <Text variant="subtext">
            {t('modelManager.source')}: {data?.source}
          </Text>
        )}
        <Box mt="4">
          <ModelAttrView label="Description" value={data.description} />
        </Box>
      </Flex>

      <Tabs mt="4" h="100%">
        <TabList>
          <Tab>{t('modelManager.settings')}</Tab>
          <Tab>{t('modelManager.metadata')}</Tab>
        </TabList>

        <TabPanels h="100%">
          <TabPanel>{selectedModelMode === 'view' ? <ModelView /> : <ModelEdit />}</TabPanel>
          <TabPanel h="full">
            <ModelMetadata />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </>
  );
};
