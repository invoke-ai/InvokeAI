/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { FilterOperationSessionState } from '@workbench/widgets/layers/filterOperationSession';

import { Box, Heading, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, MenuContent } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import {
  buildFilterDefaults,
  getFilterDefinition,
  isFilterConfigValid,
} from '@workbench/generation/canvas/filterGraphs';
import { useFilterSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { LayerFilterControls } from '@workbench/widgets/layers/LayerFilterControls';
import { ChevronDownIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { CanvasOperationPanel } from './CanvasOperationPanel';

export interface FilterActionEligibility {
  canApply: boolean;
  canCancel: boolean;
  canEdit: boolean;
  canProcess: boolean;
  canReset: boolean;
  canSave: boolean;
}

export const getFilterSaveTargetEligibility = (
  eligibility: Pick<FilterActionEligibility, 'canSave'>
): Record<'raster' | 'control', boolean> => ({
  control: eligibility.canSave,
  raster: eligibility.canSave,
});

export const getFilterActionEligibility = (
  session: FilterOperationSessionState,
  isExternalInteractionLocked = false
): FilterActionEligibility => {
  const busy = session.status === 'processing' || session.status === 'committing';
  const actionsBlocked = busy || isExternalInteractionLocked;
  const hasPreview = session.preview !== null && !actionsBlocked;
  const isValid = isFilterConfigValid(session.draft.type, session.draft.settings);
  return {
    canApply: hasPreview,
    canCancel: true,
    canEdit: !actionsBlocked,
    canProcess: !actionsBlocked && isValid,
    canReset: !actionsBlocked,
    canSave: hasPreview,
  };
};

export const FilterPanelHeading = ({
  layerName,
  layerTypeLabel,
  title,
}: {
  layerName: string;
  layerTypeLabel: string;
  title: string;
}) => (
  <Box>
    <Heading fontSize="md" id="filter-operation-title">
      {title}
    </Heading>
    <Text color="fg.muted" fontSize="xs">
      {layerName} · {layerTypeLabel}
    </Text>
  </Box>
);

const FilterFeedback = ({ error, status }: Pick<FilterOperationSessionState, 'error' | 'status'>) => {
  const { t } = useTranslation();
  const statusText =
    status === 'processing'
      ? t('widgets.layers.rasterFilter.running')
      : status === 'committing'
        ? t('widgets.layers.rasterFilter.statusCommitting')
        : status === 'error'
          ? t('widgets.layers.rasterFilter.statusError')
          : t('widgets.layers.selectObject.statusReady');
  return (
    <Stack fontSize="xs" gap="1">
      <Text aria-live="polite" color="fg.muted" role="status">
        {statusText}
      </Text>
      {error ? (
        <Text aria-live="assertive" color="fg.error" role="alert">
          {error}
        </Text>
      ) : null}
    </Stack>
  );
};

export const FilterOptions = ({
  engine,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & { isExternalInteractionLocked?: boolean }) => {
  const { t } = useTranslation();
  const session = useFilterSession(engine);
  if (!session) {
    return null;
  }

  const eligibility = getFilterActionEligibility(session, isExternalInteractionLocked);
  const saveTargetEligibility = getFilterSaveTargetEligibility(eligibility);
  const setType = (type: string) => {
    const definition = getFilterDefinition(type);
    engine.updateFilterOperation({ settings: definition ? buildFilterDefaults(definition) : {}, type });
  };
  const setSettings = (settings: Record<string, unknown>) => {
    const current = engine.stores.filterSession.get();
    if (current) {
      engine.updateFilterOperation({ settings, type: current.draft.type });
    }
  };
  const reset = () => {
    const current = engine.stores.filterSession.get();
    if (!current) {
      return;
    }
    const definition = getFilterDefinition(current.draft.type);
    engine.resetFilterOperation(definition ? buildFilterDefaults(definition) : {});
  };

  return (
    <CanvasOperationPanel.Root aria-labelledby="filter-operation-title" operation="filter">
      <CanvasOperationPanel.Header>
        <FilterPanelHeading
          layerName={session.layerName}
          layerTypeLabel={t(`widgets.layers.selectObject.saveAs_${session.layerType}`)}
          title={t('widgets.layers.rasterFilter.title')}
        />
      </CanvasOperationPanel.Header>
      <CanvasOperationPanel.Body>
        <Stack gap="4">
          <LayerFilterControls
            disabled={!eligibility.canEdit}
            filterType={session.draft.type}
            focusFilter={false}
            settings={session.draft.settings}
            variant="operation"
            onFilterTypeChange={setType}
            onSettingsChange={setSettings}
          />
        </Stack>
      </CanvasOperationPanel.Body>
      <CanvasOperationPanel.Feedback>
        <FilterFeedback error={session.error} status={session.status} />
      </CanvasOperationPanel.Feedback>
      <CanvasOperationPanel.Footer>
        <Button
          disabled={!eligibility.canProcess}
          loading={session.status === 'processing'}
          minH="10"
          onClick={() => void engine.processFilterOperation()}
        >
          {t('widgets.layers.selectObject.process')}
        </Button>
        <Button disabled={!eligibility.canReset} minH="10" variant="outline" onClick={reset}>
          {t('widgets.layers.selectObject.reset')}
        </Button>
        <Button
          disabled={!eligibility.canApply}
          loading={session.status === 'committing'}
          minH="10"
          onClick={() => void engine.commitFilterOperation('apply', makeImageDurable)}
        >
          {t('common.apply')}
        </Button>
        <Menu.Root>
          <Menu.Trigger asChild>
            <Button disabled={!eligibility.canSave} minH="10" variant="outline">
              {t('widgets.layers.selectObject.saveAs')} <ChevronDownIcon size={14} />
            </Button>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="11rem" py="1">
                <Menu.Item
                  disabled={!saveTargetEligibility.raster}
                  value="raster"
                  onClick={() => void engine.commitFilterOperation('raster', makeImageDurable)}
                >
                  <Menu.ItemText>{t('widgets.layers.selectObject.saveAs_raster')}</Menu.ItemText>
                </Menu.Item>
                <Menu.Item
                  disabled={!saveTargetEligibility.control}
                  value="control"
                  onClick={() => void engine.commitFilterOperation('control', makeImageDurable)}
                >
                  <Menu.ItemText>{t('widgets.layers.selectObject.saveAs_control')}</Menu.ItemText>
                </Menu.Item>
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
        <Button
          disabled={!eligibility.canCancel}
          minH="10"
          variant="ghost"
          onClick={() => engine.cancelFilterOperation()}
        >
          {t('common.cancel')}
        </Button>
      </CanvasOperationPanel.Footer>
    </CanvasOperationPanel.Root>
  );
};
