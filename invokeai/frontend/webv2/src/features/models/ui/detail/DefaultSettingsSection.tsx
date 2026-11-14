/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { AnyModelDefaultSettings, ModelConfig } from '@features/models/core/types';
import type { DefaultComponentSlotPolicy, ModelDefaultsRulesState } from '@features/models/ui/ModelsUiContext';

import { createListCollection, Grid, HStack, Icon, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { hasModelPickerCandidates } from '@features/models/core/library';
import { updateModel } from '@features/models/data/api';
import {
  ensureFp8StorageSupportLoaded,
  isFp8StorageSupported,
  useFp8StorageSupportSelector,
} from '@features/models/data/fp8StorageSupportStore';
import { replaceModelInStore, useModelsSelector } from '@features/models/data/modelsStore';
import { ModelSelect } from '@features/models/ui/components/ModelSelect';
import {
  applyModelDraftFields,
  beginModelDraftSave,
  discardModelDraft,
  failModelDraftSave,
  finishModelDraftSave,
  hasDraftFields,
  hasModelDraftConflict,
  recordModelDraftFields,
  useModelDraft,
  type ModelDraft,
} from '@features/models/ui/modelDraftsStore';
import { useModelsUi } from '@features/models/ui/ModelsUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, Combobox, FieldLabel, Panel, Select } from '@platform/ui';
import { Field } from '@platform/ui/Field';
import { MoveHorizontalIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import type { DefaultSettingsControl, DefaultSettingsModel, SchedulerFieldRules } from './defaultSettingsFields';

import { getFieldsForModel, validateDefaults } from './defaultSettingsFields';
import { DraftConflictNotice, UnsavedChangesLabel } from './ModelDraftNotices';

/**
 * Disabled defaults persist null to inherit app settings; defaultSettingsFields owns field policy and validation.
 * Unsaved values live in the model's retained draft, overlaid on the current server settings.
 */

const EMPTY_SETTINGS: AnyModelDefaultSettings = {};
const EMPTY_COMPONENTS: Readonly<Record<string, string>> = {};
const EMPTY_CATALOG: readonly ModelConfig[] = [];
const EMPTY_SLOTS: readonly DefaultComponentSlotPolicy[] = [];
const selectDefaultsDraft = (draft: ModelDraft | undefined) => draft?.defaults;

interface FieldControlProps {
  /** Generate's VAE rule for the edited model; absent until the rules arrive. */
  acceptsVae: ((candidate: ModelConfig) => boolean) | null;
  control: DefaultSettingsControl;
  disabled: boolean;
  label: string;
  /** Base of the model being edited, for same-base model pickers. */
  modelBase: string;
  /** Key of the model being edited; its links rank the model pickers. */
  modelKey: string;
  setValue: (value: unknown) => void;
  value: unknown;
}

// Built per option list; memoized by the rendering component per field key.
const buildCollection = (options: readonly string[]) =>
  createListCollection({ items: options.map((option) => ({ label: option, value: option })) });

const FieldControl = ({
  acceptsVae,
  control,
  disabled,
  label,
  modelBase,
  modelKey,
  setValue,
  value,
}: FieldControlProps) => {
  const { t } = useTranslation();
  const selectCollection = useMemo(
    () => (control.kind === 'select' ? buildCollection(control.options) : null),
    [control]
  );

  if (control.kind === 'number') {
    return (
      <NumberInput.Root
        disabled={disabled}
        max={control.max}
        min={control.min}
        position="relative"
        size="lg"
        step={control.step ?? 1}
        value={typeof value === 'number' ? String(value) : ''}
        w="full"
        onValueChange={(details) => {
          // Empty/partial input is transient; only commit finite numbers so the
          // field never silently flips back to "off".
          if (Number.isFinite(details.valueAsNumber)) {
            setValue(details.valueAsNumber);
          }
        }}
      >
        <NumberInput.Control />
        {/* Drag horizontally on the handle to scrub the value. */}
        <NumberInput.Scrubber
          alignItems="center"
          bottom="0"
          cursor="ew-resize"
          display="flex"
          left="2"
          position="absolute"
          top="0"
          zIndex={1}
        >
          <Icon as={MoveHorizontalIcon} boxSize="3" color="fg.subtle" />
        </NumberInput.Scrubber>
        <NumberInput.Input aria-label={label} ps="7" />
      </NumberInput.Root>
    );
  }

  if (control.kind === 'select' && selectCollection) {
    return (
      <Select
        aria-label={label}
        collection={selectCollection}
        disabled={disabled}
        size="lg"
        value={typeof value === 'string' ? [value] : []}
        onValueChange={({ value: next }) => {
          const nextValue = next[0];

          if (nextValue !== undefined) {
            setValue(nextValue);
          }
        }}
      />
    );
  }

  if (control.kind === 'combobox') {
    return (
      <Combobox
        aria-label={label}
        disabled={disabled}
        options={control.options}
        size="lg"
        value={typeof value === 'string' ? value : null}
        onValueChange={setValue}
      />
    );
  }

  if (control.kind === 'model') {
    return (
      <ModelSelect
        disabled={disabled}
        filter={(candidate) =>
          !control.generateVaeRule || (acceptsVae ? acceptsVae(candidate) : String(candidate.base) === modelBase)
        }
        isClearable
        modelTypes={control.modelTypes}
        placeholder={t(control.placeholderKey)}
        relatedTo={modelKey}
        showManagerButton={false}
        size="lg"
        // The 'default' sentinel renders as the placeholder: the toggle is on
        // but no specific model is pinned.
        value={typeof value === 'string' && value !== 'default' ? value : null}
        onChange={(selected) => setValue(selected ? selected.key : 'default')}
      />
    );
  }

  return null;
};

export const DefaultSettingsSection = ({
  model,
  onError,
  onSaved,
}: {
  model: DefaultSettingsModel;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  // Whether FP8 Storage reaches this model depends on its format and on its loader, so the backend
  // answers it. Fetched here because this section is the only thing that asks, and the table is static
  // per backend build. While it is unknown the row stays hidden: a control that arrives a moment late
  // is better than one that does nothing.
  useMountEffect(() => {
    void ensureFp8StorageSupportLoaded();
  });
  const fp8StorageSupported = useFp8StorageSupportSelector((snapshot) => isFp8StorageSupported(snapshot, model));
  // Generate's scheduler rules load with the section, as the FP8 answer does; the row shows disabled until they do.
  const { defaultsRules: defaultsRulesSource } = useModelsUi();
  useMountEffect(() => {
    defaultsRulesSource.load();
  });
  const defaultsRulesState = useSyncExternalStore(
    defaultsRulesSource.subscribe,
    defaultsRulesSource.getSnapshot,
    defaultsRulesSource.getSnapshot
  );
  const schedulerRules = useMemo((): SchedulerFieldRules | null => {
    if (model.type !== 'main') {
      return null;
    }
    return defaultsRulesState.status === 'ready'
      ? defaultsRulesState.read(model, model.default_settings?.scheduler ?? null)
      : defaultsRulesState.status;
  }, [model, defaultsRulesState]);
  // The failure a retry started from. The retry is over once the rules arrive or a new failure replaces it, so a later
  // ordinary load (another account, say) shows the plain loading text rather than "Retrying…".
  const [retriedFailure, setRetriedFailure] = useState<ModelDefaultsRulesState | null>(null);
  if (
    retriedFailure &&
    (defaultsRulesState.status === 'ready' ||
      (defaultsRulesState.status === 'unavailable' && defaultsRulesState !== retriedFailure))
  ) {
    setRetriedFailure(null);
  }
  const isRetryingDefaultsRules = retriedFailure !== null;
  const schedulerSwitchRef = useRef<HTMLInputElement | null>(null);
  const headingRef = useRef<HTMLDivElement | null>(null);
  const retryDefaultsRules = () => {
    if (defaultsRulesState.status !== 'unavailable') {
      return;
    }
    setRetriedFailure(defaultsRulesState);
    defaultsRulesSource.load();
  };
  // When a retry succeeds the Retry button leaves with the note; hand the focus it held to the scheduler row, or to
  // the section heading when the base ignores the scheduler and the whole row left with it.
  const handOverRetryFocus = useCallback((button: HTMLButtonElement | null) => {
    if (!button) {
      return undefined;
    }
    return () => {
      if (button !== document.activeElement) {
        return;
      }
      // After the commit, once the row has either stayed or gone.
      queueMicrotask(() => {
        const schedulerSwitch = schedulerSwitchRef.current;
        (schedulerSwitch?.isConnected ? schedulerSwitch : headingRef.current)?.focus();
      });
    };
  }, []);
  const acceptsVae = useMemo(
    () =>
      defaultsRulesState.status === 'ready'
        ? (candidate: ModelConfig) => defaultsRulesState.acceptsDefaultVae(model, candidate)
        : null,
    [defaultsRulesState, model]
  );
  const fields = useMemo(
    () => getFieldsForModel(model, fp8StorageSupported, schedulerRules),
    [fp8StorageSupported, model, schedulerRules]
  );
  const saved = model.default_settings ?? EMPTY_SETTINGS;
  const draft = useModelDraft(model.key, selectDefaultsDraft);
  const settings = useMemo(() => applyModelDraftFields(saved, draft), [draft, saved]);
  const [error, setError] = useState<string | null>(null);
  const { run } = useScopedAction();
  // From the draft, not this instance: a save started before the user navigated away may still be in flight.
  const isSaving = draft?.submitted !== undefined;
  const isDirty = hasDraftFields(draft);

  const setFieldValue = (key: keyof AnyModelDefaultSettings, value: unknown) => {
    recordModelDraftFields(model.key, 'defaults', { ...settings, [key]: value }, saved);
    setError(null);
  };

  // The component slots of the model's family, which depend on the components chosen so far. Only a main model has
  // any, so the others do not follow the catalog.
  const isMain = model.type === 'main';
  const catalog = useModelsSelector((snapshot) => (isMain ? snapshot.models : EMPTY_CATALOG));
  const isCatalogLoaded = useModelsSelector((snapshot) => snapshot.status === 'loaded');
  const storedComponents = settings.components ?? EMPTY_COMPONENTS;
  const componentSlots = useMemo(
    () =>
      isMain && defaultsRulesState.status === 'ready'
        ? defaultsRulesState.readComponentSlots(model, storedComponents, catalog)
        : null,
    [catalog, defaultsRulesState, isMain, model, storedComponents]
  );
  // The group shows once the rules are ready. A slot nothing installed can fill stays out of the way, unless it holds
  // a value (even a dangling one), which must stay clearable. The status of the rules is the scheduler row's to show.
  const visibleComponentSlots = useMemo(
    () =>
      (componentSlots ?? EMPTY_SLOTS).filter(
        (slot) =>
          storedComponents[slot.persistenceKey] !== undefined ||
          !isCatalogLoaded ||
          hasModelPickerCandidates(catalog, { filter: slot.filter, modelTypes: slot.modelTypes })
      ),
    [catalog, componentSlots, isCatalogLoaded, storedComponents]
  );

  // An empty map is stored as null, which the backend and the draft read as "no defaults".
  const setComponent = (persistenceKey: string, modelKey: string | null) => {
    if (defaultsRulesState.status !== 'ready') {
      return;
    }

    const next = { ...storedComponents };

    if (modelKey) {
      next[persistenceKey] = modelKey;
    } else {
      delete next[persistenceKey];
    }

    // A slot this change closes (the vision tower once its encoder is cleared) leaves no hidden default behind. A key
    // for a slot not offered before the change stays: another version or client may know slots this one does not.
    const before = new Set(componentSlots?.map((slot) => slot.persistenceKey));
    const after = new Set(
      defaultsRulesState.readComponentSlots(model, next, catalog).map((slot) => slot.persistenceKey)
    );

    for (const key of Object.keys(next)) {
      if (before.has(key) && !after.has(key)) {
        delete next[key];
      }
    }

    setFieldValue('components', Object.keys(next).length > 0 ? next : null);
  };

  const handleReset = () => {
    discardModelDraft(model.key, 'defaults');
    setError(null);
  };

  const handleSave = async () => {
    const validationError = validateDefaults(model, settings, t);

    if (validationError) {
      setError(validationError);
      return;
    }

    const submitted = settings;

    await run(
      async (owner) => {
        if (!beginModelDraftSave(model.key, 'defaults', submitted)) {
          return;
        }

        const updated = await updateModel(model.key, { default_settings: submitted }, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        finishModelDraftSave(model.key, 'defaults', updated.default_settings ?? EMPTY_SETTINGS);
        onSaved();
      },
      (_message, saveError) => {
        failModelDraftSave(model.key, 'defaults', saved, null);
        onError(saveError instanceof Error ? saveError.message : t('models.failedToSaveDefaults'));
      }
    );
  };

  return (
    <Stack gap="3">
      <HStack justify="space-between">
        <Stack ref={headingRef} gap="0.5" tabIndex={-1}>
          <FieldLabel>{t('models.defaultSettings')}</FieldLabel>
          <Text color="fg.subtle" fontSize="xs">
            {t('models.defaultSettingsHelp')}
          </Text>
        </Stack>
        <HStack flexShrink={0} gap="2">
          {isDirty ? (
            <>
              <UnsavedChangesLabel />
              <Button disabled={isSaving} variant="ghost" onClick={handleReset}>
                {t('common.reset')}
              </Button>
            </>
          ) : null}
          <Button disabled={!isDirty} loading={isSaving} variant="solid" onClick={() => void handleSave()}>
            {t('models.saveDefaults')}
          </Button>
        </HStack>
      </HStack>
      {error ? (
        <Text color="fg.error" fontSize="xs" role="alert">
          {error}
        </Text>
      ) : null}
      {hasModelDraftConflict(saved, draft) ? <DraftConflictNotice /> : null}
      <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
        {fields.map((field) => {
          const value = settings[field.key] ?? null;
          const isEnabled = value !== null && value !== undefined;

          return (
            <Panel key={field.key} gap="2" p="2.5" tone="surface">
              <HStack justify="space-between">
                <Text fontSize="xs" fontWeight="600" textTransform="uppercase">
                  {t(field.labelKey)}
                </Text>
                <Switch.Root
                  checked={isEnabled}
                  colorPalette="accent"
                  // Without the rules there is no default to turn on, but a stored value can still be cleared.
                  disabled={field.pending !== undefined && !isEnabled}
                  size="sm"
                  onCheckedChange={(event) => setFieldValue(field.key, event.checked ? field.defaultValue : null)}
                >
                  <Switch.HiddenInput ref={field.key === 'scheduler' ? schedulerSwitchRef : undefined} />
                  <Switch.Control>
                    <Switch.Thumb />
                  </Switch.Control>
                  <Switch.Label srOnly>{t('models.customizeDefaultField', { field: t(field.labelKey) })}</Switch.Label>
                </Switch.Root>
              </HStack>
              {field.control ? (
                <FieldControl
                  control={field.control}
                  disabled={!isEnabled || field.pending !== undefined}
                  label={t(field.labelKey)}
                  acceptsVae={acceptsVae}
                  modelBase={String(model.base)}
                  modelKey={model.key}
                  setValue={(nextValue) => setFieldValue(field.key, nextValue)}
                  value={isEnabled ? value : field.defaultValue}
                />
              ) : null}
              {field.pending === 'unavailable' || (field.pending === 'loading' && isRetryingDefaultsRules) ? (
                // One Retry button across the failure and the retry's load, so the focus it holds stays put.
                <HStack gap="2" justify="space-between" role={field.pending === 'unavailable' ? 'alert' : 'status'}>
                  <Text color={field.pending === 'unavailable' ? 'fg.warning' : 'fg.subtle'} fontSize="xs">
                    {field.pending === 'unavailable'
                      ? t('models.schedulerListUnavailable')
                      : t('models.schedulerListLoading')}
                  </Text>
                  <Button
                    ref={handOverRetryFocus}
                    aria-busy={field.pending === 'loading'}
                    // `aria-disabled` rather than `disabled`: a disabled button drops the focus it holds.
                    aria-disabled={field.pending === 'loading'}
                    size="xs"
                    variant="outline"
                    onClick={retryDefaultsRules}
                  >
                    {field.pending === 'loading' ? t('models.schedulerListRetrying') : t('common.retry')}
                  </Button>
                </HStack>
              ) : (
                <Text color="fg.subtle" fontSize="xs" role={field.pending === 'loading' ? 'status' : undefined}>
                  {field.pending === 'loading'
                    ? t('models.schedulerListLoading')
                    : isEnabled
                      ? t('models.customizedForThisModel')
                      : t(field.inheritLabelKey)}
                </Text>
              )}
            </Panel>
          );
        })}
      </Grid>
      {visibleComponentSlots.length > 0 ? (
        <Stack gap="2">
          <Stack gap="0.5">
            <FieldLabel>{t('models.defaultComponents')}</FieldLabel>
            <Text color="fg.subtle" fontSize="xs">
              {t('models.defaultComponentsHelp')}
            </Text>
          </Stack>
          <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
            {visibleComponentSlots.map((slot) => (
              <Panel key={slot.persistenceKey} gap="2" p="2.5" tone="surface">
                <Field helpText={slot.helpText} label={slot.label}>
                  <ModelSelect
                    filter={slot.filter}
                    id={`default-component-${slot.persistenceKey}`}
                    isClearable
                    modelTypes={slot.modelTypes}
                    placeholder={t('models.defaultComponentsNone')}
                    relatedTo={model.key}
                    showManagerButton={false}
                    size="lg"
                    value={storedComponents[slot.persistenceKey] ?? null}
                    onChange={(selected) => setComponent(slot.persistenceKey, selected?.key ?? null)}
                  />
                </Field>
              </Panel>
            ))}
          </Grid>
        </Stack>
      ) : null}
    </Stack>
  );
};
