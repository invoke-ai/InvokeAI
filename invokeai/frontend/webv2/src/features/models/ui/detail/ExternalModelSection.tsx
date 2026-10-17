/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ExternalModelSettingsModel } from '@features/models/core/externalModelFields';
import type { ExternalModelFormValues } from '@features/models/core/schemas';

import {
  Checkbox,
  createListCollection,
  Grid,
  HStack,
  Input,
  NumberInput,
  Stack,
  Switch,
  Text,
} from '@chakra-ui/react';
import {
  isStarterManagedExternalModel,
  toExternalModelChanges,
  toExternalModelValues,
  validateExternalModel,
} from '@features/models/core/externalModelFields';
import { updateModel } from '@features/models/data/api';
import {
  ensureExternalProvidersLoaded,
  useExternalProvidersSelector,
} from '@features/models/data/externalProvidersStore';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { ensureStartersLoaded, refreshStarters, useStartersSelector } from '@features/models/data/startersStore';
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
import { useMountEffect } from '@platform/react/useMountEffect';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { Field, FieldLabel } from '@platform/ui/Field';
import { Panel } from '@platform/ui/Panel';
import { Select } from '@platform/ui/Select';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { DraftConflictNotice, UnsavedChangesLabel } from './ModelDraftNotices';

/**
 * Edits the hosted-provider record: which provider and model ID requests route to, the modes and inputs the model
 * advertises, its limits, and the size/count the provider is asked for by default. A record the starter catalog
 * manages stays read-only: the backend overlays the catalog entry matched by `source` on every response, so a
 * changed route would keep the original entry's capabilities. Unsaved values live in the model's retained draft;
 * `externalModelFields` owns the record mapping and validation.
 */

const selectExternalDraft = (draft: ModelDraft | undefined) => draft?.external;

type OptionalNumberKey = {
  [Key in keyof ExternalModelFormValues]: ExternalModelFormValues[Key] extends number | null ? Key : never;
}[keyof ExternalModelFormValues];
type FlagKey = {
  [Key in keyof ExternalModelFormValues]: ExternalModelFormValues[Key] extends boolean ? Key : never;
}[keyof ExternalModelFormValues];

const MODE_FLAGS: readonly { key: FlagKey; labelKey: string }[] = [
  { key: 'modeTxt2img', labelKey: 'models.externalModes.txt2img' },
  { key: 'modeImg2img', labelKey: 'models.externalModes.img2img' },
  { key: 'modeInpaint', labelKey: 'models.externalModes.inpaint' },
];

const SUPPORT_FLAGS: readonly { key: FlagKey; labelKey: string }[] = [
  { key: 'supportsSeed', labelKey: 'models.externalFields.supportsSeed' },
  { key: 'supportsReferenceImages', labelKey: 'models.externalFields.supportsReferenceImages' },
  { key: 'supportsNegativePrompt', labelKey: 'models.externalFields.supportsNegativePrompt' },
];

const LIMIT_FIELDS: readonly { key: OptionalNumberKey; labelKey: string }[] = [
  { key: 'maxImagesPerRequest', labelKey: 'models.externalFields.maxImagesPerRequest' },
  { key: 'maxReferenceImages', labelKey: 'models.externalFields.maxReferenceImages' },
  { key: 'maxImageWidth', labelKey: 'models.externalFields.maxImageWidth' },
  { key: 'maxImageHeight', labelKey: 'models.externalFields.maxImageHeight' },
];

const DEFAULT_FIELDS: readonly { key: OptionalNumberKey; labelKey: string }[] = [
  { key: 'defaultWidth', labelKey: 'models.externalFields.defaultWidth' },
  { key: 'defaultHeight', labelKey: 'models.externalFields.defaultHeight' },
  { key: 'defaultNumImages', labelKey: 'models.externalFields.defaultNumImages' },
];

const ReadOnlyValue = ({ label, value }: { label: string; value: string }) => (
  <Stack gap="0.5">
    <Text color="fg.subtle" fontSize="xs" fontWeight="600" textTransform="uppercase">
      {label}
    </Text>
    <Text fontSize="sm">{value}</Text>
  </Stack>
);

const OptionalNumberField = ({
  disabled,
  label,
  placeholder,
  setValue,
  value,
}: {
  disabled: boolean;
  label: string;
  placeholder: string;
  setValue: (value: number | null) => void;
  value: number | null;
}) => (
  <Field label={label}>
    <NumberInput.Root
      disabled={disabled}
      min={1}
      size="lg"
      step={1}
      value={value === null ? '' : String(value)}
      w="full"
      onValueChange={(details) => {
        // Clearing the field lifts the limit; a partial entry is transient and never commits.
        if (details.value === '') {
          setValue(null);
        } else if (Number.isFinite(details.valueAsNumber)) {
          setValue(details.valueAsNumber);
        }
      }}
    >
      <NumberInput.Control />
      <NumberInput.Input aria-label={label} placeholder={placeholder} />
    </NumberInput.Root>
  </Field>
);

export const ExternalModelSection = ({
  model,
  onError,
  onSaved,
}: {
  model: ExternalModelSettingsModel;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  const saved = useMemo(() => toExternalModelValues(model), [model]);
  const draft = useModelDraft(model.key, selectExternalDraft);
  const values = useMemo(() => applyModelDraftFields(saved, draft), [draft, saved]);
  const [error, setError] = useState<string | null>(null);
  const { run } = useScopedAction();
  const providerConfigs = useExternalProvidersSelector((snapshot) => snapshot.configs);
  const starters = useStartersSelector((snapshot) => snapshot.response?.starter_models ?? null);
  const startersStatus = useStartersSelector((snapshot) => snapshot.status);
  // A failed lookup cannot establish that edits persist: the backend may still overlay catalog metadata.
  const isStarterManaged = starters !== null && isStarterManagedExternalModel(model, starters);
  const canEdit = starters !== null && !isStarterManaged;
  // The registered providers are what requests can route to; a record's own id stays offered even when it is not
  // among them, so an unknown provider reads as what it is rather than as nothing.
  const providerCollection = useMemo(
    () =>
      providerConfigs === null
        ? null
        : createListCollection({
            items: [
              ...new Set([...providerConfigs.map((config) => config.provider_id), saved.providerId, values.providerId]),
            ]
              .filter((id) => id !== '')
              .map((id) => ({ label: id, value: id })),
          }),
    [providerConfigs, saved.providerId, values.providerId]
  );
  // From the draft, not this instance: a save started before the user navigated away may still be in flight.
  const isSaving = draft?.submitted !== undefined;
  const isDirty = hasDraftFields(draft);

  useMountEffect(() => {
    ensureStartersLoaded();
    // The list is a convenience: while it is unavailable the provider stays a text field.
    ensureExternalProvidersLoaded().catch(() => {});
  });

  const setField = <Key extends keyof ExternalModelFormValues>(key: Key, value: ExternalModelFormValues[Key]) => {
    recordModelDraftFields(model.key, 'external', { ...values, [key]: value }, saved);
    setError(null);
  };

  const handleReset = () => {
    discardModelDraft(model.key, 'external');
    setError(null);
  };

  const handleSave = async () => {
    if (!canEdit) {
      return;
    }

    const validationError = validateExternalModel(values);

    if (validationError) {
      setError(validationError);
      return;
    }

    const submitted = values;

    await run(
      async (owner) => {
        if (!beginModelDraftSave(model.key, 'external', submitted)) {
          return;
        }

        const updated = await updateModel(model.key, toExternalModelChanges(submitted, model), owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        finishModelDraftSave(model.key, 'external', toExternalModelValues(updated));
        onSaved();
      },
      (_message, saveError) => {
        failModelDraftSave(model.key, 'external', saved, null);
        onError(saveError instanceof Error ? saveError.message : t('models.failedToSaveExternalSettings'));
      }
    );
  };

  const numberField = ({ key, labelKey }: { key: OptionalNumberKey; labelKey: string }, placeholderKey: string) => (
    <OptionalNumberField
      key={key}
      disabled={isSaving}
      label={t(labelKey)}
      placeholder={t(placeholderKey)}
      setValue={(value) => setField(key, value)}
      value={values[key]}
    />
  );

  return (
    <Stack gap="3">
      <HStack justify="space-between">
        <Stack gap="0.5">
          <FieldLabel>{t('models.externalProvider')}</FieldLabel>
          <Text color="fg.subtle" fontSize="xs">
            {t('models.externalProviderHelp')}
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
          <Button disabled={!canEdit || !isDirty} loading={isSaving} variant="solid" onClick={() => void handleSave()}>
            {t('models.saveExternalSettings')}
          </Button>
        </HStack>
      </HStack>
      {error ? (
        <Text color="fg.error" fontSize="xs" role="alert">
          {error}
        </Text>
      ) : null}
      {hasModelDraftConflict(saved, draft) ? <DraftConflictNotice /> : null}
      {starters === null ? (
        <Panel gap="2" p="2.5" role={startersStatus === 'error' ? 'alert' : 'status'} tone="surface">
          <Text color={startersStatus === 'error' ? 'fg.error' : 'fg.muted'} fontSize="xs">
            {t(
              startersStatus === 'error'
                ? 'models.externalFields.catalogUnavailable'
                : 'models.externalFields.catalogLoading'
            )}
          </Text>
          {startersStatus === 'error' ? (
            <Button alignSelf="start" variant="outline" onClick={() => void refreshStarters()}>
              {t('common.retry')}
            </Button>
          ) : null}
        </Panel>
      ) : null}
      {!canEdit && isDirty ? (
        <Text color="fg.muted" fontSize="xs" role="note">
          {t('models.externalFields.lockedDraftBlocked')}
        </Text>
      ) : null}
      <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
        <Field helpText={t('models.providerIdHelp')} label={t('models.providerId')}>
          {providerCollection ? (
            <Select
              aria-label={t('models.providerId')}
              collection={providerCollection}
              disabled={!canEdit || isSaving}
              size="lg"
              value={values.providerId === '' ? [] : [values.providerId]}
              onValueChange={({ value }) => {
                if (value[0] !== undefined) {
                  setField('providerId', value[0]);
                }
              }}
            />
          ) : (
            <Input
              aria-label={t('models.providerId')}
              disabled={!canEdit || isSaving}
              size="lg"
              value={values.providerId}
              onChange={(event) => setField('providerId', event.currentTarget.value)}
            />
          )}
        </Field>
        <Field helpText={t('models.providerModelIdHelp')} label={t('models.providerModelId')}>
          <Input
            disabled={!canEdit || isSaving}
            size="lg"
            value={values.providerModelId}
            onChange={(event) => setField('providerModelId', event.currentTarget.value)}
          />
        </Field>
      </Grid>
      {!canEdit ? (
        <Panel gap="2.5" p="2.5" tone="surface">
          {starters === null ? null : (
            <Text color="fg.muted" fontSize="xs" role="note">
              {t('models.externalFields.starterManaged')}
            </Text>
          )}
          <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
            <ReadOnlyValue
              label={t('models.externalFields.supportedModes')}
              value={MODE_FLAGS.filter(({ key }) => values[key])
                .map(({ labelKey }) => t(labelKey))
                .join(', ')}
            />
            {SUPPORT_FLAGS.map(({ key, labelKey }) => (
              <ReadOnlyValue
                key={key}
                label={t(labelKey)}
                value={t(values[key] ? 'models.externalFields.supported' : 'models.externalFields.notSupported')}
              />
            ))}
            {LIMIT_FIELDS.map(({ key, labelKey }) => (
              <ReadOnlyValue
                key={key}
                label={t(labelKey)}
                value={values[key] === null ? t('models.externalFields.unlimited') : String(values[key])}
              />
            ))}
            {DEFAULT_FIELDS.map(({ key, labelKey }) => (
              <ReadOnlyValue
                key={key}
                label={t(labelKey)}
                value={values[key] === null ? t('models.externalFields.providerDefault') : String(values[key])}
              />
            ))}
          </Grid>
        </Panel>
      ) : (
        <>
          <Panel gap="2" p="2.5" tone="surface">
            <Text fontSize="xs" fontWeight="600" textTransform="uppercase">
              {t('models.externalFields.supportedModes')}
            </Text>
            <HStack gap="4" wrap="wrap">
              {MODE_FLAGS.map(({ key, labelKey }) => (
                <Checkbox.Root
                  key={key}
                  checked={values[key]}
                  disabled={isSaving}
                  size="sm"
                  onCheckedChange={(event) => setField(key, event.checked === true)}
                >
                  <Checkbox.HiddenInput />
                  <Checkbox.Control />
                  <Checkbox.Label fontSize="xs">{t(labelKey)}</Checkbox.Label>
                </Checkbox.Root>
              ))}
            </HStack>
            <Text color="fg.subtle" fontSize="xs">
              {t('models.externalFields.supportedModesHelp')}
            </Text>
          </Panel>
          <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
            {SUPPORT_FLAGS.map(({ key, labelKey }) => (
              <Panel key={key} gap="2" p="2.5" tone="surface">
                <HStack justify="space-between">
                  <Text fontSize="xs" fontWeight="600" textTransform="uppercase">
                    {t(labelKey)}
                  </Text>
                  <Switch.Root
                    checked={values[key]}
                    colorPalette="accent"
                    disabled={isSaving}
                    size="sm"
                    onCheckedChange={(event) => setField(key, event.checked)}
                  >
                    <Switch.HiddenInput />
                    <Switch.Control>
                      <Switch.Thumb />
                    </Switch.Control>
                    <Switch.Label srOnly>{t(labelKey)}</Switch.Label>
                  </Switch.Root>
                </HStack>
              </Panel>
            ))}
          </Grid>
          <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
            {LIMIT_FIELDS.map((field) => numberField(field, 'models.externalFields.unlimited'))}
          </Grid>
          <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
            {DEFAULT_FIELDS.map((field) => numberField(field, 'models.externalFields.providerDefault'))}
          </Grid>
        </>
      )}
    </Stack>
  );
};
