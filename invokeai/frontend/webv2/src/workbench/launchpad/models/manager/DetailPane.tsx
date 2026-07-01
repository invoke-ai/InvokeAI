/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { Scrollable, Tabs } from '@workbench/components/ui';
import { AddModelsView } from '@workbench/launchpad/models/add-models/AddModelsView';
import { ApiKeysSection } from '@workbench/launchpad/models/credentials/ApiKeysSection';
import { ModelDetail } from '@workbench/launchpad/models/detail/ModelDetail';
import { InstallQueueBar } from '@workbench/launchpad/models/install-queue/InstallQueueBar';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { updateModelsUi, useModelsUiSelector, type ModelManagerTab } from '@workbench/models/uiStore';
import { BoxIcon, KeyRoundIcon, PlusIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { HEADER_MIN_HEIGHT } from './layoutConstants';

/** The tabbed detail pane: selected model, Add Models, API Keys, and queue footer. */
export const DetailPane = () => {
  const { t } = useTranslation();
  const { activeModelKey, activeTab } = useModelsUiSelector(
    (snapshot) => ({ activeModelKey: snapshot.activeModelKey, activeTab: snapshot.activeTab }),
    (left, right) => left.activeModelKey === right.activeModelKey && left.activeTab === right.activeTab
  );
  const detailLabel = useModelsSelector(
    (snapshot) => snapshot.models.find((model) => model.key === activeModelKey)?.name ?? t('models.details')
  );

  return (
    <Flex direction="column" flex="1" minH="0" minW="0">
      <Flex align="flex-end" borderBottomWidth={1} flexShrink={0} minH={HEADER_MIN_HEIGHT} px="2">
        <Tabs.Root
          size="sm"
          mb="-1px"
          value={activeTab}
          onValueChange={(event) => updateModelsUi({ activeTab: event.value as ModelManagerTab })}
        >
          <Tabs.List h="full">
            <Tabs.Trigger fontSize="xs" value="details">
              <Icon as={BoxIcon} boxSize="3" />
              <Text maxW="14rem" truncate>
                {detailLabel}
              </Text>
            </Tabs.Trigger>
            <Tabs.Trigger fontSize="xs" value="add">
              <Icon as={PlusIcon} boxSize="3" />
              {t('models.addModels')}
            </Tabs.Trigger>
            <Tabs.Trigger fontSize="xs" value="keys">
              <Icon as={KeyRoundIcon} boxSize="3" />
              {t('models.apiKeys')}
            </Tabs.Trigger>
          </Tabs.List>
        </Tabs.Root>
      </Flex>

      <Box flex="1" minH="0">
        {activeTab === 'details' ? <DetailTab modelKey={activeModelKey} /> : null}
        {activeTab === 'add' ? <AddModelsView /> : null}
        {activeTab === 'keys' ? (
          <Scrollable h="full" label={t('models.apiKeys')} minH="0" p="3">
            <ApiKeysSection />
          </Scrollable>
        ) : null}
      </Box>

      <InstallQueueBar />
    </Flex>
  );
};

const DetailTab = ({ modelKey }: { modelKey: string | null }) => {
  const { t } = useTranslation();
  const handleDeleted = useCallback(() => updateModelsUi({ activeModelKey: null }), []);

  if (modelKey === null) {
    return (
      <Flex align="center" direction="column" gap="2" h="full" justify="center" p="6">
        <Icon as={BoxIcon} boxSize="8" color="fg.subtle" />
        <Text color="fg.muted" fontSize="sm" fontWeight="600">
          {t('models.selectModel')}
        </Text>
        <Text color="fg.subtle" fontSize="xs" maxW="22rem" textAlign="center">
          {t('models.selectModelDescription')}
        </Text>
      </Flex>
    );
  }

  return (
    <Scrollable h="full" label={t('models.details')} minH="0" p="3">
      <ModelDetail key={modelKey} density="full" modelKey={modelKey} onDeleted={handleDeleted} />
    </Scrollable>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
