import type {
  OperationPropertyForm,
  ToolFormProps,
  ToolFooterProps,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Box, createListCollection, Flex, Icon, Menu, Portal } from '@chakra-ui/react';
import { galleryDurability } from '@features/gallery';
import { Button } from '@platform/ui/Button';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Select } from '@platform/ui/Select';
import {
  buildFilterDefaults,
  CONTROL_FILTERS,
  FILTER_CATEGORY_ORDER,
  getCanvasOperations,
  getFilterDefinition,
  isFilterConfigValid,
  recordLastUsedFilterType,
  type FilterOperationSessionState,
  type FilterSessionErrorCode,
} from '@workbench/canvas-operations/api';
import { useFilterSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertySwitchRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { LayerFilterControls } from '@workbench/widgets/layers/LayerFilterControls';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { OperationStatusChip } from './OperationStatusSlot';

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

const FILTER_ERROR_TRANSLATION_KEYS: Record<FilterSessionErrorCode, string> = {
  aborted: 'widgets.layers.rasterFilter.aborted',
  'apply-failed': 'widgets.layers.rasterFilter.applyFailed',
  busy: 'widgets.layers.rasterFilter.busy',
  disabled: 'widgets.layers.rasterFilter.disabled',
  empty: 'widgets.layers.rasterFilter.empty',
  locked: 'widgets.layers.rasterFilter.locked',
  missing: 'widgets.layers.rasterFilter.missing',
  'not-ready': 'widgets.layers.rasterFilter.notReady',
  'over-budget': 'widgets.layers.rasterFilter.overBudget',
  'process-failed': 'widgets.layers.rasterFilter.processFailed',
  'source-over-budget': 'widgets.layers.rasterFilter.sourceOverBudget',
  stale: 'widgets.layers.rasterFilter.stale',
  unsupported: 'widgets.layers.rasterFilter.unsupported',
};

export const getFilterErrorTranslationKey = (code: FilterSessionErrorCode): string =>
  FILTER_ERROR_TRANSLATION_KEYS[code];

const useFilterDraft = (engine: ToolFormProps['engine']) => {
  const operations = getCanvasOperations(engine);
  const setType = useCallback(
    (type: string) => {
      const definition = getFilterDefinition(type);
      recordLastUsedFilterType(type);
      operations.updateFilterOperation({ settings: definition ? buildFilterDefaults(definition) : {}, type });
    },
    [operations]
  );
  const setSettings = useCallback(
    (settings: Record<string, unknown>) => {
      const current = operations.getFilterSessionState();
      if (current) {
        operations.updateFilterOperation({ settings, type: current.draft.type });
      }
    },
    [operations]
  );
  const reset = useCallback(() => {
    const current = operations.getFilterSessionState();
    if (current) {
      const definition = getFilterDefinition(current.draft.type);
      operations.resetFilterOperation(definition ? buildFilterDefaults(definition) : {});
    }
  }, [operations]);
  return { operations, reset, setSettings, setType };
};

/** Filter choice (grouped by category) and auto-process: the iteration dials. */
const FilterChooseSettings = ({ engine, isSurfaceInteractionLocked }: ToolFormProps) => {
  const { t } = useTranslation();
  const session = useFilterSession(engine);
  const { operations, setType } = useFilterDraft(engine);
  const filterCollection = useMemo(() => {
    const order = new Map<string, number>(FILTER_CATEGORY_ORDER.map((category, index) => [category, index]));
    const items = [...CONTROL_FILTERS]
      .sort(
        (a, b) =>
          (order.get(a.category) ?? 0) - (order.get(b.category) ?? 0) ||
          t(`widgets.layers.control.filters.${a.type}`, a.type).localeCompare(
            t(`widgets.layers.control.filters.${b.type}`, b.type)
          )
      )
      .map((filter) => ({
        category: filter.category,
        label: t(`widgets.layers.control.filters.${filter.type}`, filter.type),
        value: filter.type,
      }));
    return createListCollection({ items });
  }, [t]);
  const filterValue = useMemo(() => [session?.draft.type ?? ''], [session?.draft.type]);
  const groupBy = useCallback((item: { category: string }) => item.category, []);
  const renderGroupLabel = useCallback(
    (category: string) => t(`widgets.layers.rasterFilter.categories.${category}`, category),
    [t]
  );
  const onTypeChange = useCallback(
    ({ value }: { value: string[] }) => {
      if (value[0]) {
        setType(value[0]);
      }
    },
    [setType]
  );
  if (!session) {
    return null;
  }
  const eligibility = getFilterActionEligibility(session, isSurfaceInteractionLocked);
  return (
    <>
      <Select
        aria-label={t('widgets.layers.control.filter')}
        collection={filterCollection}
        disabled={!eligibility.canEdit}
        groupBy={groupBy}
        itemsMaxH="18rem"
        renderGroupLabel={renderGroupLabel}
        size="xs"
        value={filterValue}
        valueText={t(`widgets.layers.control.filters.${session.draft.type}`, session.draft.type)}
        w="full"
        onValueChange={onTypeChange}
      />
      <PropertySwitchRow
        checked={session.autoProcess}
        disabled={!eligibility.canEdit}
        label={t('widgets.layers.rasterFilter.autoProcess')}
        onCheckedChange={(checked) => operations.setFilterOperationAutoProcess(checked)}
      />
    </>
  );
};

/** The chosen filter's parameters. */
const FilterParamsSettings = ({ engine, isSurfaceInteractionLocked }: ToolFormProps) => {
  const session = useFilterSession(engine);
  const { setSettings, setType } = useFilterDraft(engine);
  if (!session) {
    return null;
  }
  const eligibility = getFilterActionEligibility(session, isSurfaceInteractionLocked);
  return (
    <LayerFilterControls
      disabled={!eligibility.canEdit}
      filterType={session.draft.type}
      focusFilter={false}
      parts="params"
      settings={session.draft.settings}
      onFilterTypeChange={setType}
      onSettingsChange={setSettings}
    />
  );
};

const FilterFooter = ({ engine, isExternalInteractionLocked }: ToolFooterProps) => {
  const { t } = useTranslation();
  const session = useFilterSession(engine);
  const { operations, reset } = useFilterDraft(engine);
  const onApply = useCallback(
    () => void operations.commitFilterOperation('apply', galleryDurability.makeCanvasAsset),
    [operations]
  );
  const onCancel = useCallback(() => operations.cancelFilterOperation(), [operations]);
  if (!session) {
    return null;
  }
  const eligibility = getFilterActionEligibility(session, isExternalInteractionLocked);
  const saveTargets = getFilterSaveTargetEligibility(eligibility);
  const sourceLabel = `${session.layerName} · ${t(`widgets.layers.selectObject.saveAs_${session.layerType}`)}`;
  const isBusy = !session.error && (session.status === 'processing' || session.status === 'committing');
  return (
    <Flex align="center" gap="1" minW="0" w="full">
      <Box flex="1" minW="0" overflow="hidden" whiteSpace="nowrap">
        <OperationStatusChip
          errorDetail={session.error?.detail ?? null}
          errorText={session.error ? t(getFilterErrorTranslationKey(session.error.code)) : null}
          isBusy={isBusy}
          sourceLabel={sourceLabel}
          statusText={t(getFilterStatusTranslationKey(session.status))}
          technicalDetailsLabel={t('widgets.layers.selectObject.technicalDetails')}
          title={t('widgets.layers.rasterFilter.title')}
        />
      </Box>
      <Button disabled={!eligibility.canReset} flexShrink={0} size="xs" variant="ghost" onClick={reset}>
        {t('widgets.layers.selectObject.reset')}
      </Button>
      {session.autoProcess ? null : (
        <Button
          disabled={!eligibility.canProcess}
          flexShrink={0}
          loading={session.status === 'processing'}
          size="xs"
          onClick={() => void operations.processFilterOperation()}
        >
          {t('widgets.layers.selectObject.process')}
        </Button>
      )}
      <Flex flexShrink={0}>
        <Button
          data-pane-action="apply"
          disabled={!eligibility.canApply}
          loading={session.status === 'committing'}
          roundedRight="none"
          size="xs"
          variant="solid"
          onClick={onApply}
        >
          {t('common.apply')}
        </Button>
        <Menu.Root positioning={APPLY_MENU_POSITIONING}>
          <Menu.Trigger asChild>
            <Button
              aria-label={t('widgets.layers.rasterFilter.applyAs')}
              disabled={!eligibility.canSave}
              px="1"
              roundedLeft="none"
              size="xs"
              variant="solid"
            >
              <Icon as={ChevronDownIcon} boxSize="3.5" />
            </Button>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="12rem" py="1">
                <MenuActionItem
                  disabled={!saveTargets.raster}
                  label={t('widgets.layers.selectObject.saveAs_raster')}
                  value="save-raster"
                  onSelect={() => void operations.commitFilterOperation('raster', galleryDurability.makeCanvasAsset)}
                />
                <MenuActionItem
                  disabled={!saveTargets.control}
                  label={t('widgets.layers.selectObject.saveAs_control')}
                  value="save-control"
                  onSelect={() => void operations.commitFilterOperation('control', galleryDurability.makeCanvasAsset)}
                />
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Flex>
      <Button
        data-pane-action="cancel"
        disabled={!eligibility.canCancel}
        flexShrink={0}
        size="xs"
        variant="ghost"
        onClick={onCancel}
      >
        {t('common.cancel')}
      </Button>
    </Flex>
  );
};

const APPLY_MENU_POSITIONING = { placement: 'top-end' } as const;

export const filterOperationForm: OperationPropertyForm = {
  footer: FilterFooter,
  groups: [
    { body: FilterChooseSettings, id: 'filter-choose', labelKey: 'widgets.layers.control.filter' },
    { body: FilterParamsSettings, id: 'filter-params', labelKey: 'widgets.properties.groups.settings' },
  ],
  kind: 'filter',
};
