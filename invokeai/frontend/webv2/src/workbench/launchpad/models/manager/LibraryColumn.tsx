/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, Flex, HStack, Icon, Separator, Text } from '@chakra-ui/react';
import { Button, IconButton, ConfirmDialog } from '@workbench/components/ui';
import { MaintenanceMenu } from '@workbench/launchpad/models/library/MaintenanceMenu';
import { ModelFilterBar } from '@workbench/launchpad/models/library/ModelFilterBar';
import { ModelLibraryList } from '@workbench/launchpad/models/library/ModelLibraryList';
import { bulkDeleteModels } from '@workbench/models/api';
import { collectBases, collectTypes } from '@workbench/models/library';
import { refreshModels, removeModelsFromStore, useModelsSelector } from '@workbench/models/modelsStore';
import {
  openModelDetail,
  pruneModelsUiKeys,
  toggleModelSelection,
  updateModelsUi,
  useModelsUiSelector,
} from '@workbench/models/uiStore';
import { useNotify } from '@workbench/useNotify';
import { Trash2Icon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';

import { HEADER_MIN_HEIGHT, LIBRARY_WIDTH } from './layoutConstants';

/** The persistent master list: header, search/filter bar, and bulk actions. */
export const LibraryColumn = () => {
  const notify = useNotify();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const missingCount = useModelsSelector((snapshot) => snapshot.missingModelKeys.size);
  const { activeModelKey, filters, selectedKeys } = useModelsUiSelector(
    (snapshot) => ({
      activeModelKey: snapshot.activeModelKey,
      filters: snapshot.filters,
      selectedKeys: snapshot.selectedKeys,
    }),
    (left, right) =>
      left.activeModelKey === right.activeModelKey &&
      left.filters === right.filters &&
      left.selectedKeys === right.selectedKeys
  );
  const [isBulkDeleteOpen, setIsBulkDeleteOpen] = useState(false);

  const availableTypes = useMemo(() => collectTypes(models), [models]);
  const availableBases = useMemo(() => collectBases(models), [models]);
  const handleActivate = useCallback((modelKey: string) => openModelDetail(modelKey), []);
  const handleToggleSelected = useCallback((modelKey: string) => toggleModelSelection(modelKey), []);

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
    <Flex direction="column" flexShrink={0} h="full" minH="0" position="relative" w={LIBRARY_WIDTH} borderEndWidth={1}>
      <HStack align="center" borderBottomWidth={1} flexShrink={0} gap="2" minH={HEADER_MIN_HEIGHT} px="3">
        <Text fontSize="sm" fontWeight="700">
          Models
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {models.length}
        </Text>
        <Box ms="auto">
          <MaintenanceMenu />
        </Box>
      </HStack>

      <ModelFilterBar
        availableBases={availableBases}
        availableTypes={availableTypes}
        filters={filters}
        missingCount={missingCount}
        onChange={(nextFilters) => updateModelsUi({ filters: nextFilters })}
      />

      <ModelLibraryList
        activeModelKey={activeModelKey}
        filters={filters}
        instanceId="manager"
        selectedKeys={selectedKeys}
        onActivate={handleActivate}
        onToggleSelected={handleToggleSelected}
      />

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
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
