import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { Scrollable, Tabs } from '@workbench/components/ui';
import { AddModelsView } from '@workbench/launchpad/models/add-models/AddModelsView';
import { ApiKeysSection } from '@workbench/launchpad/models/credentials/ApiKeysSection';
import { ModelDetail } from '@workbench/launchpad/models/detail/ModelDetail';
import { InstallQueueBar } from '@workbench/launchpad/models/install-queue/InstallQueueBar';
import { useModelsSnapshot } from '@workbench/models/modelsStore';
import { updateModelsUi, useModelsUi, type ModelManagerTab } from '@workbench/models/uiStore';
import { BoxIcon, KeyRoundIcon, PlusIcon } from 'lucide-react';

import { HEADER_MIN_HEIGHT } from './layoutConstants';

/** The tabbed detail pane: selected model, Add Models, API Keys, and queue footer. */
export const DetailPane = () => {
  const { activeModelKey, activeTab } = useModelsUi();
  const { models } = useModelsSnapshot();

  const activeModel = activeModelKey ? models.find((model) => model.key === activeModelKey) : undefined;
  const detailLabel = activeModel?.name ?? 'Model Details';

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
              Add Models
            </Tabs.Trigger>
            <Tabs.Trigger fontSize="xs" value="keys">
              <Icon as={KeyRoundIcon} boxSize="3" />
              API Keys
            </Tabs.Trigger>
          </Tabs.List>
        </Tabs.Root>
      </Flex>

      <Box flex="1" minH="0">
        {activeTab === 'details' ? <DetailTab modelKey={activeModelKey} /> : null}
        {activeTab === 'add' ? <AddModelsView /> : null}
        {activeTab === 'keys' ? (
          <Scrollable h="full" label="API keys" minH="0" p="3">
            <ApiKeysSection />
          </Scrollable>
        ) : null}
      </Box>

      <InstallQueueBar />
    </Flex>
  );
};

const DetailTab = ({ modelKey }: { modelKey: string | null }) => {
  if (modelKey === null) {
    return (
      <Flex align="center" direction="column" gap="2" h="full" justify="center" p="6">
        <Icon as={BoxIcon} boxSize="8" color="fg.subtle" />
        <Text color="fg.muted" fontSize="sm" fontWeight="600">
          Select a model
        </Text>
        <Text color="fg.subtle" fontSize="xs" maxW="22rem" textAlign="center">
          Pick a model from the library to view details, edit metadata, set per-model defaults, and manage trigger
          phrases.
        </Text>
      </Flex>
    );
  }

  return (
    <Scrollable h="full" label="Model details" minH="0" p="3">
      <ModelDetail
        key={modelKey}
        density="full"
        modelKey={modelKey}
        onDeleted={() => updateModelsUi({ activeModelKey: null })}
      />
    </Scrollable>
  );
};
