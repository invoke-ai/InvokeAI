import type {
  OperationPropertyForm,
  OperationStatus,
  ToolFormProps,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import { createListCollection } from '@chakra-ui/react';
import { galleryDurability } from '@features/gallery';
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
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { OperationActions } from './OperationActions';

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

export const getFilterActionEligibility = (session: FilterOperationSessionState): FilterActionEligibility => {
  const actionsBlocked = session.status === 'processing' || session.status === 'committing';
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

/** The phase shown while the filter works; null at rest. */
export const getFilterBusyTranslationKey = (status: FilterOperationSessionState['status']): string | null =>
  status === 'processing'
    ? 'widgets.layers.rasterFilter.running'
    : status === 'committing'
      ? 'widgets.layers.rasterFilter.statusCommitting'
      : null;

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
  'source-adjustment-error': 'widgets.layers.rasterFilter.sourceAdjustmentError',
  'source-adjustment-over-budget': 'widgets.layers.rasterFilter.sourceAdjustmentOverBudget',
  'source-adjustment-too-large': 'widgets.layers.rasterFilter.sourceAdjustmentTooLarge',
  'source-over-budget': 'widgets.layers.rasterFilter.sourceOverBudget',
  'source-unreadable': 'widgets.layers.rasterFilter.sourceUnreadable',
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

/** Filter choice (grouped by category) and auto-update: the iteration dials. */
const FilterChooseSettings = ({ engine }: ToolFormProps) => {
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
  const eligibility = getFilterActionEligibility(session);
  return (
    <>
      <Select
        aria-label={t('widgets.layers.control.filter')}
        collection={filterCollection}
        disabled={!eligibility.canEdit}
        groupBy={groupBy}
        itemsMaxH="18rem"
        renderGroupLabel={renderGroupLabel}
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
const FilterParamsSettings = ({ engine }: ToolFormProps) => {
  const session = useFilterSession(engine);
  const { setSettings, setType } = useFilterDraft(engine);
  if (!session) {
    return null;
  }
  const eligibility = getFilterActionEligibility(session);
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

const useFilterStatus = (engine: ToolFormProps['engine']): OperationStatus | null => {
  const { t } = useTranslation();
  const session = useFilterSession(engine);
  if (!session) {
    return null;
  }
  const busyKey = getFilterBusyTranslationKey(session.status);
  return {
    busyText: busyKey ? t(busyKey) : null,
    error: session.error
      ? { detail: session.error.detail ?? null, text: t(getFilterErrorTranslationKey(session.error.code)) }
      : null,
    source: `${session.layerName} · ${t(`widgets.layers.selectObject.saveAs_${session.layerType}`)}`,
  };
};

const FilterActions = ({ engine }: ToolFormProps) => {
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
  const eligibility = getFilterActionEligibility(session);
  const saveTargets = getFilterSaveTargetEligibility(eligibility);
  return (
    <OperationActions
      applying={session.status === 'committing'}
      canApply={eligibility.canApply}
      canCancel={eligibility.canCancel}
      canReset={eligibility.canReset}
      canSave={eligibility.canSave}
      run={
        session.autoProcess
          ? null
          : {
              disabled: !eligibility.canProcess,
              label: t('widgets.layers.rasterFilter.process'),
              loading: session.status === 'processing',
              onRun: () => void operations.processFilterOperation(),
            }
      }
      saveAsLabel={t('widgets.layers.rasterFilter.applyAs')}
      saveTargets={[
        {
          disabled: !saveTargets.raster,
          label: t('widgets.layers.selectObject.saveAs_raster'),
          value: 'save-raster',
          onSelect: () => void operations.commitFilterOperation('raster', galleryDurability.makeCanvasAsset),
        },
        {
          disabled: !saveTargets.control,
          label: t('widgets.layers.selectObject.saveAs_control'),
          value: 'save-control',
          onSelect: () => void operations.commitFilterOperation('control', galleryDurability.makeCanvasAsset),
        },
      ]}
      onApply={onApply}
      onCancel={onCancel}
      onReset={reset}
    />
  );
};

export const filterOperationForm: OperationPropertyForm = {
  actions: FilterActions,
  groups: [
    { body: FilterChooseSettings, id: 'filter-choose', labelKey: 'widgets.layers.control.filter' },
    { body: FilterParamsSettings, id: 'filter-params', labelKey: 'widgets.properties.groups.settings' },
  ],
  kind: 'filter',
  useStatus: useFilterStatus,
};
