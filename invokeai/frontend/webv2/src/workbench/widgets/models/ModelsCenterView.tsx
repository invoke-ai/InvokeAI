import { Badge, Box, Flex, HStack, Icon, Separator, Stack, Text } from '@chakra-ui/react';
import { BoxIcon, Trash2Icon, XIcon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { Button, IconButton } from '../../components/ui/Button';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { Scrollable } from '../../components/ui/Scrollable';
import { Tabs } from '../../components/ui/Tabs';
import { bulkDeleteModels } from '../../models/api';
import { ensureInstallsLoaded, isActiveInstallStatus, useInstallsSnapshot } from '../../models/installsStore';
import { collectBases, collectTypes } from '../../models/library';
import { ensureModelsLoaded, refreshModels, removeModelsFromStore, useModelsSnapshot } from '../../models/modelsStore';
import {
  pruneModelsUiKeys,
  toggleModelSelection,
  updateModelsUi,
  useModelsUi,
  type ModelsCenterTab,
} from '../../models/uiStore';
import { useNotify } from '../../useNotify';
import { AddModelsView } from './AddModelsView';
import { InstallQueueSection } from './InstallQueueSection';
import { ModelDetail } from './ModelDetail';
import { ModelFilterBar } from './ModelFilterBar';
import { ModelLibraryList } from './ModelLibraryList';

/**
 * Full model manager: Library (master-detail with bulk actions), Add Models
 * (every install source), and the install Queue with a live activity badge.
 * Tab choice, selection, and filters live in the models UI store so nothing
 * resets when the user moves between tabs or away from the center view.
 */
export const ModelsCenterView = () => {
  const { centerTab } = useModelsUi();
  const { jobs } = useInstallsSnapshot();
  const activeInstallCount = jobs.filter((job) => isActiveInstallStatus(job.status)).length;

  useEffect(() => {
    ensureModelsLoaded();
    ensureInstallsLoaded();
  }, []);

  return (
    <Flex direction="column" gap="3" h="full" minH="0" w="full" px="3">
      <Tabs.Root
        size="sm"
        value={centerTab}
        onValueChange={(event) => updateModelsUi({ centerTab: event.value as ModelsCenterTab })}
      >
        <Tabs.List>
          <Tabs.Trigger fontSize="xs" value="library">
            Library
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="add">
            Add Models
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="queue">
            Queue
            {activeInstallCount > 0 ? (
              <Badge colorPalette="blue" fontSize="2xs" ms="1" size="sm" variant="solid">
                {activeInstallCount}
              </Badge>
            ) : null}
          </Tabs.Trigger>
        </Tabs.List>
      </Tabs.Root>
      <Box flex="1" minH="0">
        {centerTab === 'library' ? <LibraryMasterDetail /> : null}
        {centerTab === 'add' ? <AddModelsView /> : null}
        {centerTab === 'queue' ? (
          <Box h="full" maxW="48rem" minH="0">
            <InstallQueueSection />
          </Box>
        ) : null}
      </Box>
    </Flex>
  );
};

const LibraryMasterDetail = () => {
  const notify = useNotify();
  const { missingModelKeys, models } = useModelsSnapshot();
  const { activeModelKey, filters, selectedKeys } = useModelsUi();
  const [isBulkDeleteOpen, setIsBulkDeleteOpen] = useState(false);

  const availableTypes = useMemo(() => collectTypes(models), [models]);
  const availableBases = useMemo(() => collectBases(models), [models]);

  const handleBulkDelete = async () => {
    const keys = [...selectedKeys];

    try {
      const result = await bulkDeleteModels(keys);

      removeModelsFromStore(result.deleted);
      pruneModelsUiKeys(result.deleted);
      updateModelsUi({ selectedKeys: new Set(result.failed.map((failure) => failure.key)) });

      if (result.failed.length > 0) {
        notify.error(
          'Some models could not be deleted',
          `${result.deleted.length} deleted; ${result.failed.length} failed: ${result.failed[0]?.error ?? ''}`
        );
      } else {
        notify.success(
          'Models deleted',
          `${result.deleted.length} model${result.deleted.length === 1 ? '' : 's'} deleted.`
        );
      }
    } catch (error) {
      notify.error('Bulk delete failed', error instanceof Error ? error.message : String(error));
      void refreshModels();
    }
  };

  return (
    <Flex h="full" minH="0" w="full" gap="2">
      <Stack flexShrink={0} gap="2" h="full" minH="0" position="relative" w="22rem">
        <ModelFilterBar
          availableBases={availableBases}
          availableTypes={availableTypes}
          filters={filters}
          missingCount={missingModelKeys.size}
          onChange={(nextFilters) => updateModelsUi({ filters: nextFilters })}
        />
        <ModelLibraryList
          activeModelKey={activeModelKey}
          filters={filters}
          instanceId="center"
          selectedKeys={selectedKeys}
          onActivate={(model) => updateModelsUi({ activeModelKey: model.key })}
          onToggleSelected={(model) => toggleModelSelection(model.key)}
        />
        {/* ActionBar-styled, but plain markup: dialog-based action bars
            dismiss on outside interaction, which would clear the selection on
            every additional click. Floats over the list, never shifts it. */}
        {selectedKeys.size > 0 ? (
          <HStack
            bg="bg.muted"
            borderColor="border.emphasized"
            borderWidth="1px"
            bottom="2"
            gap="2"
            insetX="2"
            p="1.5"
            position="absolute"
            rounded="lg"
            shadow="lg"
            zIndex={2}
          >
            <Text
              borderColor="border.subtle"
              borderStyle="dashed"
              borderWidth="1px"
              color="fg.muted"
              fontSize="2xs"
              fontWeight="600"
              px="2"
              py="1"
              rounded="md"
            >
              {selectedKeys.size} selected
            </Text>
            <Separator borderColor="border.subtle" h="5" orientation="vertical" />
            <Button colorPalette="red" size="2xs" variant="solid" onClick={() => setIsBulkDeleteOpen(true)}>
              <Icon as={Trash2Icon} boxSize="3" />
              Delete
            </Button>
            <Box flex="1" />
            <IconButton
              aria-label="Clear selection"
              size="2xs"
              variant="ghost"
              onClick={() => updateModelsUi({ selectedKeys: new Set() })}
            >
              <Icon as={XIcon} boxSize="3" />
            </IconButton>
          </HStack>
        ) : null}
      </Stack>
      <Scrollable flex="1" h="full" label="Model details" minH="0" minW="0" ps="4" pe="1" borderStartWidth={1}>
        {activeModelKey !== null ? (
          <ModelDetail
            key={activeModelKey}
            density="full"
            modelKey={activeModelKey}
            onDeleted={() => updateModelsUi({ activeModelKey: null })}
          />
        ) : (
          <Flex align="center" direction="column" gap="2" h="full" justify="center">
            <Icon as={BoxIcon} boxSize="8" color="fg.subtle" />
            <Text color="fg.muted" fontSize="sm" fontWeight="600">
              Select a model
            </Text>
            <Text color="fg.subtle" fontSize="xs" maxW="22rem" textAlign="center">
              View details, edit metadata, set per-model defaults, and manage trigger phrases.
            </Text>
          </Flex>
        )}
      </Scrollable>
      <ConfirmDialog
        body={`Delete ${selectedKeys.size} selected model${selectedKeys.size === 1 ? '' : 's'}? Database records are removed, and files inside the InvokeAI models directory are deleted.`}
        confirmLabel={`Delete ${selectedKeys.size} Model${selectedKeys.size === 1 ? '' : 's'}`}
        isOpen={isBulkDeleteOpen}
        title="Delete selected models"
        onClose={() => setIsBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
      />
    </Flex>
  );
};
