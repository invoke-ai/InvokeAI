import type { WidgetId } from '@workbench/types';

import { Box, HStack, Icon, Progress, Stack, Text } from '@chakra-ui/react';
import { Button, Scrollable, Tooltip } from '@workbench/components/ui';
import {
  ensureInstallsLoaded,
  isActiveInstallStatus,
  useInstallProgress,
  useInstallsSnapshot,
} from '@workbench/models/installsStore';
import { collectBases, collectTypes } from '@workbench/models/library';
import { ensureModelsLoaded, useModelsSnapshot } from '@workbench/models/modelsStore';
import { openModelsCenterTab, updateModelsUi, useModelsUi } from '@workbench/models/uiStore';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { DownloadIcon, PlusIcon } from 'lucide-react';
import { useEffect, useMemo } from 'react';

import { ModelDetail } from './ModelDetail';
import { ModelFilterBar } from './ModelFilterBar';
import { ModelLibraryList } from './ModelLibraryList';

/**
 * Side-panel presentation: a searchable library browser with in-panel
 * drill-in for quick edits. Installing and bulk work live in the full center
 * view; the Add button jumps straight to its Add Models tab, the install
 * footer to its Queue tab. Filters and the drilled-in model persist in the
 * models UI store across panel/tab switches.
 */
export const ModelsPanelView = ({ widgetId }: { widgetId: WidgetId }) => {
  const { missingModelKeys, models } = useModelsSnapshot();
  const { filters, panelModelKey } = useModelsUi();
  const openWorkbenchWidget = useOpenWorkbenchWidget();

  useEffect(() => {
    ensureModelsLoaded();
    ensureInstallsLoaded();
  }, []);

  const availableTypes = useMemo(() => collectTypes(models), [models]);
  const availableBases = useMemo(() => collectBases(models), [models]);

  return (
    <Box h="full" minH="0" position="relative">
      {/* Drill-in renders as an overlay so the list below keeps its layout —
          and therefore its scroll position — for the trip back. */}
      {panelModelKey !== null ? (
        <Scrollable bg="bg" inset="0" label="Model details" position="absolute" px="3" zIndex={2}>
          <ModelDetail
            key={panelModelKey}
            density="panel"
            modelKey={panelModelKey}
            onBack={() => updateModelsUi({ panelModelKey: null })}
            onDeleted={() => updateModelsUi({ panelModelKey: null })}
          />
        </Scrollable>
      ) : null}
      {/* The list stays mounted under the overlay (preserving layout and
          scroll position) but is inert so it cannot scroll or take focus. */}
      <Stack gap="2" h="full" inert={panelModelKey !== null} minH="0" px="2">
        <HStack gap="1.5">
          <Box flex="1" minW="0">
            <ModelFilterBar
              availableBases={availableBases}
              availableTypes={availableTypes}
              filters={filters}
              missingCount={missingModelKeys.size}
              onChange={(nextFilters) => updateModelsUi({ filters: nextFilters })}
            />
          </Box>
          <Tooltip content="Add models">
            <Button
              aria-label="Add models"
              flexShrink={0}
              size="sm"
              variant="solid"
              onClick={() => {
                openModelsCenterTab('add');
                openWorkbenchWidget(widgetId, { preferredRegions: ['center'], requireCenterView: true });
              }}
            >
              <Icon as={PlusIcon} boxSize="3.5" />
              Add
            </Button>
          </Tooltip>
        </HStack>
        <ModelLibraryList
          activeModelKey={null}
          filters={filters}
          instanceId="panel"
          onActivate={(model) => updateModelsUi({ panelModelKey: model.key })}
        />
        <InstallSummaryFooter
          onOpen={() => {
            openModelsCenterTab('queue');
            openWorkbenchWidget(widgetId, { preferredRegions: ['center'], requireCenterView: true });
          }}
        />
      </Stack>
    </Box>
  );
};

/** Compact live footer: how many installs are running, with progress. */
const InstallSummaryFooter = ({ onOpen }: { onOpen: () => void }) => {
  const { jobs } = useInstallsSnapshot();
  const activeJobs = jobs.filter((job) => isActiveInstallStatus(job.status));
  const errorCount = jobs.filter((job) => job.status === 'error').length;
  const firstDownloading = activeJobs.find((job) => job.status === 'downloading');

  if (activeJobs.length === 0 && errorCount === 0) {
    return null;
  }

  return (
    <Stack
      as="button"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      cursor="pointer"
      flexShrink={0}
      gap="1"
      p="2"
      rounded="md"
      textAlign="start"
      w="full"
      _hover={{ bg: 'bg.muted' }}
      onClick={onOpen}
    >
      <HStack gap="1.5" justify="space-between">
        <HStack gap="1.5" minW="0">
          <Icon as={DownloadIcon} boxSize="3" color="fg.muted" />
          <Text color="fg.muted" fontSize="2xs" fontWeight="600" truncate>
            {activeJobs.length > 0
              ? `${activeJobs.length} install${activeJobs.length === 1 ? '' : 's'} in progress`
              : 'Install queue'}
          </Text>
        </HStack>
        {errorCount > 0 ? (
          <Text color="fg.error" flexShrink={0} fontSize="2xs" fontWeight="600">
            {errorCount} failed
          </Text>
        ) : null}
      </HStack>
      {firstDownloading ? <FooterDownloadProgress jobId={firstDownloading.id} /> : null}
    </Stack>
  );
};

const FooterDownloadProgress = ({ jobId }: { jobId: number }) => {
  const progress = useInstallProgress(jobId);
  const ratio = progress && progress.totalBytes > 0 ? Math.min(1, progress.bytes / progress.totalBytes) : null;

  return (
    <Progress.Root aria-label="Install download progress" max={1} size="xs" value={ratio}>
      <Progress.Track>
        <Progress.Range />
      </Progress.Track>
    </Progress.Root>
  );
};
