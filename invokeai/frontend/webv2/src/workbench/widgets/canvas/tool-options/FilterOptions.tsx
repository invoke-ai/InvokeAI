/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { FilterOperationSessionState } from '@workbench/widgets/layers/filterOperationSession';

import { Flex, Group, HStack, IconButton, Menu, Portal, Text, VisuallyHidden } from '@chakra-ui/react';
import { Button, MenuContent, Tooltip } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import {
  buildFilterDefaults,
  getFilterDefinition,
  isFilterConfigValid,
} from '@workbench/generation/canvas/filterGraphs';
import { CanvasFloatingBar, CanvasFloatingBarDivider } from '@workbench/widgets/canvas/CanvasFloatingBar';
import { useFilterSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { LayerFilterControls } from '@workbench/widgets/layers/LayerFilterControls';
import { ChevronDownIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { OperationStatusSlot } from './OperationStatusSlot';

const FILTER_UPWARD_POSITIONING = { placement: 'top-end' } as const;

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

export const getFilterStatusTranslationKey = (status: FilterOperationSessionState['status']): string =>
  status === 'processing'
    ? 'widgets.layers.rasterFilter.running'
    : status === 'committing'
      ? 'widgets.layers.rasterFilter.statusCommitting'
      : status === 'error'
        ? 'widgets.layers.rasterFilter.statusError'
        : 'widgets.layers.selectObject.statusReady';

export const FilterOptionsBar = ({
  engine,
  session,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & {
  isExternalInteractionLocked?: boolean;
  session: FilterOperationSessionState;
}) => {
  const { t } = useTranslation();
  const eligibility = getFilterActionEligibility(session, isExternalInteractionLocked);
  const saveTargetEligibility = getFilterSaveTargetEligibility(eligibility);
  const sourceLabel = `${session.layerName} · ${t(`widgets.layers.selectObject.saveAs_${session.layerType}`)}`;
  const isBusy = !session.error && (session.status === 'processing' || session.status === 'committing');
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
    <CanvasFloatingBar maxW="full">
      <Flex
        align="center"
        aria-label={t('widgets.layers.rasterFilter.title')}
        flexWrap="wrap"
        gap="1"
        minW="0"
        role="group"
      >
        <Tooltip content={sourceLabel}>
          <Text flexShrink="0" fontSize="xs" fontWeight="semibold" px="1" whiteSpace="nowrap">
            {t('widgets.layers.rasterFilter.title')}
            <VisuallyHidden>{sourceLabel}</VisuallyHidden>
          </Text>
        </Tooltip>
        <CanvasFloatingBarDivider />
        <Flex align="end" flexWrap="wrap" gap="2" minW="0">
          <LayerFilterControls
            disabled={!eligibility.canEdit}
            filterType={session.draft.type}
            focusFilter={false}
            settings={session.draft.settings}
            variant="operation"
            onFilterTypeChange={setType}
            onSettingsChange={setSettings}
          />
        </Flex>
        <CanvasFloatingBarDivider />
        <Tooltip content={t('widgets.layers.rasterFilter.autoProcessDescription')}>
          <Button
            aria-pressed={session.autoProcess}
            disabled={!eligibility.canEdit}
            size="xs"
            variant={session.autoProcess ? 'solid' : 'ghost'}
            onClick={() => engine.setFilterOperationAutoProcess(!session.autoProcess)}
          >
            {t('widgets.layers.rasterFilter.autoProcess')}
          </Button>
        </Tooltip>
        <CanvasFloatingBarDivider />
        <OperationStatusSlot
          errorDetail={null}
          errorText={session.error}
          isBusy={isBusy}
          statusText={t(getFilterStatusTranslationKey(session.status))}
          technicalDetailsLabel={t('widgets.layers.selectObject.technicalDetails')}
        />
        <CanvasFloatingBarDivider />
        <HStack flexShrink="0" gap="1">
          <Button
            disabled={!eligibility.canProcess}
            loading={session.status === 'processing'}
            size="xs"
            onClick={() => void engine.processFilterOperation()}
          >
            {t('widgets.layers.selectObject.process')}
          </Button>
          <Button disabled={!eligibility.canReset} size="xs" variant="ghost" onClick={reset}>
            {t('widgets.layers.selectObject.reset')}
          </Button>
          <Menu.Root positioning={FILTER_UPWARD_POSITIONING}>
            <Group attached>
              <Button
                colorPalette="accent"
                disabled={!eligibility.canApply}
                loading={session.status === 'committing'}
                roundedEnd="none"
                size="xs"
                onClick={() => void engine.commitFilterOperation('apply', makeImageDurable)}
              >
                {t('common.apply')}
              </Button>
              <Menu.Trigger asChild>
                <IconButton
                  aria-label={t('widgets.layers.selectObject.saveAs')}
                  colorPalette="accent"
                  disabled={!eligibility.canSave}
                  minW="0"
                  roundedStart="none"
                  size="xs"
                  w="6"
                >
                  <ChevronDownIcon />
                </IconButton>
              </Menu.Trigger>
            </Group>
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
            size="xs"
            variant="ghost"
            onClick={() => engine.cancelFilterOperation()}
          >
            {t('common.cancel')}
          </Button>
        </HStack>
      </Flex>
    </CanvasFloatingBar>
  );
};

export const FilterOptions = ({
  engine,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & { isExternalInteractionLocked?: boolean }) => {
  const session = useFilterSession(engine);
  if (!session) {
    return null;
  }
  return (
    <FilterOptionsBar engine={engine} isExternalInteractionLocked={isExternalInteractionLocked} session={session} />
  );
};
