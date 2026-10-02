/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Icon } from '@chakra-ui/react';
import { collectBases, collectTypes, filterModels } from '@features/models/core/library';
import { bulkDeleteModels, bulkReidentifyModels } from '@features/models/data/api';
import { refreshModels, removeModelsFromStore, useModelsSelector } from '@features/models/data/modelsStore';
import { refreshStartersIfLoaded } from '@features/models/data/startersStore';
import { MaintenanceMenu } from '@features/models/ui/library/MaintenanceMenu';
import { ModelFilterBar } from '@features/models/ui/library/ModelFilterBar';
import { ModelLibraryList } from '@features/models/ui/library/ModelLibraryList';
import {
  openModelDetail,
  pruneModelsUiKeys,
  toggleModelSelection,
  updateModelsUi,
  useModelsUiSelector,
} from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, IconButton, ConfirmDialog } from '@platform/ui';
import { ListSelectionBar } from '@platform/ui/list/ListSelectionBar';
import { ManagerColumn } from '@platform/ui/ManagerLayout';
import { RefreshCcwIcon, Trash2Icon, XIcon } from 'lucide-react';
import { useCallback, useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const EMPTY_KEYS: string[] = [];

/** The persistent master list: header, search/filter bar, and bulk actions. */
export const LibraryColumn = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const missingModelKeys = useModelsSelector((snapshot) => snapshot.missingModelKeys);
  const missingCount = missingModelKeys.size;
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
  const [isBulkReidentifyOpen, setIsBulkReidentifyOpen] = useState(false);
  // Separate instances: run ignores re-entry, so sharing one would let a slow
  // re-identify swallow a delete.
  const { run } = useScopedAction();
  const { run: runReidentify } = useScopedAction();

  const availableTypes = useMemo(() => collectTypes(models), [models]);
  const availableBases = useMemo(() => collectBases(models), [models]);
  const hasSelection = selectedKeys.size > 0;
  const canSelectAll = models.length > 0;
  const deferredFilters = useDeferredValue(filters);
  // Match Select all to the list's deferred filters. Avoid filtering on every keystroke when nothing is selected;
  // compute on click instead.
  const filteredKeys = useMemo(
    () =>
      hasSelection ? filterModels(models, deferredFilters, missingModelKeys).map((model) => model.key) : EMPTY_KEYS,
    [deferredFilters, hasSelection, missingModelKeys, models]
  );
  const hasUnselectedFiltered = useMemo(
    () => filteredKeys.some((key) => !selectedKeys.has(key)),
    [filteredKeys, selectedKeys]
  );
  const handleToggleSelectAll = useCallback(() => {
    if (hasSelection && !hasUnselectedFiltered) {
      updateModelsUi({ selectedKeys: new Set() });

      return;
    }

    // Compute on click because the selection memo intentionally stays empty with no selection.
    const keys = filterModels(models, filters, missingModelKeys).map((model) => model.key);

    // Union: selections made under a previous filter survive, so the delete
    // confirm always shows the true total.
    updateModelsUi({ selectedKeys: new Set([...selectedKeys, ...keys]) });
  }, [filters, hasSelection, hasUnselectedFiltered, missingModelKeys, models, selectedKeys]);
  const handleActivate = useCallback((modelKey: string) => openModelDetail(modelKey), []);
  const handleToggleSelected = useCallback((modelKey: string) => toggleModelSelection(modelKey), []);

  const handleBulkDelete = async () => {
    const keys = [...selectedKeys];

    await run(
      async (owner) => {
        const result = await bulkDeleteModels(keys, owner.signal);

        assertAccountScopeCurrent(owner);
        // The relationships store prunes itself off this library change.
        removeModelsFromStore(result.deleted);
        pruneModelsUiKeys(result.deleted);
        updateModelsUi({ selectedKeys: new Set(result.failed.map((failure) => failure.key)) });
        if (result.deleted.length > 0) {
          // A deleted starter must lose its "Installed" badge in Add Models.
          refreshStartersIfLoaded();
        }

        if (result.failed.length > 0) {
          notify.error(
            t('models.someCouldNotBeDeleted'),
            t('models.bulkDeletePartialDescription', {
              deleted: result.deleted.length,
              error: result.failed[0]?.error ?? '',
              failed: result.failed.length,
            })
          );
        } else {
          notify.success(t('models.deleted'), t('models.deletedDescription', { count: result.deleted.length }));
        }
      },
      (message) => {
        notify.error(t('models.bulkDeleteFailed'), message);
        // The scope is current when onError runs, so the default capture inside
        // refreshModels targets the same account the failed call did.
        void refreshModels();
      }
    );
  };

  const handleBulkReidentify = async () => {
    const keys = [...selectedKeys];

    await runReidentify(
      async (owner) => {
        const result = await bulkReidentifyModels(keys, owner.signal);

        assertAccountScopeCurrent(owner);
        // The endpoint returns keys only; the refreshed library carries the
        // re-detected configs.
        await refreshModels(owner);
        assertAccountScopeCurrent(owner);
        updateModelsUi({ selectedKeys: new Set(result.failed.map((failure) => failure.key)) });

        if (result.failed.length > 0) {
          notify.error(
            t('models.someCouldNotBeReidentified'),
            t('models.bulkReidentifyPartialDescription', {
              error: result.failed[0]?.error ?? '',
              failed: result.failed.length,
              succeeded: result.succeeded.length,
            })
          );
        } else {
          notify.success(
            t('models.reidentified'),
            t('models.reidentifiedDescription', { count: result.succeeded.length })
          );
        }
      },
      (message) => notify.error(t('models.bulkReidentifyFailed'), message)
    );
  };

  return (
    <ManagerColumn actions={<MaintenanceMenu />} count={models.length} title={t('models.title')}>
      <ModelFilterBar
        availableBases={availableBases}
        availableTypes={availableTypes}
        filters={filters}
        missingCount={missingCount}
        onChange={(nextFilters) => updateModelsUi({ filters: nextFilters })}
      />

      <ListSelectionBar
        // No visible matches means indeterminate, not all selected; clicking clears the hidden selection.
        checked={hasSelection ? (hasUnselectedFiltered || filteredKeys.length === 0 ? 'indeterminate' : true) : false}
        isDisabled={!canSelectAll}
        label={t('models.selectAll')}
        summary={hasSelection ? t('models.selectedCount', { count: selectedKeys.size }) : undefined}
        onCheckedChange={handleToggleSelectAll}
      >
        {hasSelection ? (
          <>
            <Button size="2xs" variant="ghost" onClick={() => setIsBulkReidentifyOpen(true)}>
              <Icon as={RefreshCcwIcon} boxSize="3" />
              {t('models.reidentifySelected')}
            </Button>
            <Button colorPalette="red" size="2xs" variant="ghost" onClick={() => setIsBulkDeleteOpen(true)}>
              <Icon as={Trash2Icon} boxSize="3" />
              {t('common.delete')}
            </Button>
            <IconButton
              aria-label={t('models.clearSelection')}
              size="2xs"
              variant="ghost"
              onClick={() => updateModelsUi({ selectedKeys: new Set() })}
            >
              <Icon as={XIcon} boxSize="3" />
            </IconButton>
          </>
        ) : null}
      </ListSelectionBar>

      <ModelLibraryList
        activeModelKey={activeModelKey}
        filters={filters}
        instanceId="manager"
        selectedKeys={selectedKeys}
        onActivate={handleActivate}
        onToggleSelected={handleToggleSelected}
      />

      <ConfirmDialog
        body={t('models.bulkDeleteBody', { count: selectedKeys.size })}
        confirmLabel={t('models.bulkDeleteConfirm', { count: selectedKeys.size })}
        isOpen={isBulkDeleteOpen}
        title={t('models.deleteSelectedTitle')}
        onClose={() => setIsBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
      />
      <ConfirmDialog
        body={t('models.bulkReidentifyBody', { count: selectedKeys.size })}
        confirmLabel={t('models.bulkReidentifyConfirm', { count: selectedKeys.size })}
        isDestructive={false}
        isOpen={isBulkReidentifyOpen}
        title={t('models.bulkReidentifyTitle')}
        onClose={() => setIsBulkReidentifyOpen(false)}
        onConfirm={handleBulkReidentify}
      />
    </ManagerColumn>
  );
};
