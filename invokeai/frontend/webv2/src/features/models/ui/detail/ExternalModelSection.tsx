/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type {
  ExternalModelSettingsModel,
  ExternalModelValidationIssue,
} from '@features/models/core/externalModelFields';
import type { ExternalModelFormValues } from '@features/models/core/schemas';

import {
  Box,
  Checkbox,
  createListCollection,
  Grid,
  HStack,
  Input,
  NumberInput,
  Spinner,
  Stack,
  Switch,
  Text,
  useFieldContext,
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
import { focusFirstOperable, focusIfUnclaimed } from '@platform/react/focusIfUnclaimed';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { Field, FieldLabel } from '@platform/ui/Field';
import { Panel } from '@platform/ui/Panel';
import { Select } from '@platform/ui/Select';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
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

/** NumberInput takes only the Field's state; its description and error message are referenced here. */
const FieldNumberInputControl = ({ label, placeholder }: { label: string; placeholder: string }) => {
  const field = useFieldContext();
  const describedBy =
    [field?.ariaDescribedby, field?.getInputProps()['aria-errormessage']].filter(Boolean).join(' ') || undefined;

  return <NumberInput.Input aria-describedby={describedBy} aria-label={label} placeholder={placeholder} />;
};

const OptionalNumberField = ({
  disabled,
  error,
  label,
  placeholder,
  setValue,
  value,
}: {
  disabled: boolean;
  error: string | null;
  label: string;
  placeholder: string;
  setValue: (value: number | null) => void;
  value: number | null;
}) => (
  <Field error={error} label={label}>
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
      <FieldNumberInputControl label={label} placeholder={placeholder} />
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
  const [error, setError] = useState<ExternalModelValidationIssue | null>(null);
  const errorFor = (field: keyof ExternalModelFormValues) => (error?.field === field ? error.message : null);
  // The modes are one group; the schema reports a missing mode against the first flag.
  const modesError = errorFor('modeTxt2img');
  const modesHeadingId = useId();
  const modesMessageId = useId();
  const sectionRef = useRef<HTMLDivElement | null>(null);
  const starterManagedNoteRef = useRef<HTMLParagraphElement | null>(null);
  const { run } = useScopedAction();
  const providerConfigs = useExternalProvidersSelector((snapshot) => snapshot.configs);
  const starters = useStartersSelector((snapshot) => snapshot.response?.starter_models ?? null);
  const startersStatus = useStartersSelector((snapshot) => snapshot.status);
  const [hasRequestedRetry, setHasRequestedRetry] = useState(false);
  // The Retry button stays mounted through a retry, so it keeps the focus it holds.
  const isRetryingCatalog = hasRequestedRetry && startersStatus === 'loading';
  const showsCatalogRetry = startersStatus === 'error' || isRetryingCatalog;
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

    const validationIssue = validateExternalModel(values);

    if (validationIssue) {
      // Commit the marking first so focus can move to the field it names, scrolling it into view.
      flushSync(() => setError(validationIssue));
      focusFirstOperable(sectionRef.current?.querySelector<HTMLElement>('[data-invalid]') ?? null);
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
        onError(getApiErrorMessage(saveError, t('models.failedToSaveExternalSettings')));
      }
    );
  };

  const handleRetryCatalog = () => {
    if (isRetryingCatalog) {
      return;
    }

    setHasRequestedRetry(true);
    void refreshStarters().then(() => setHasRequestedRetry(false));
  };

  // A successful retry removes the button that held focus: hand it to the editor the catalog unlocked, or to the
  // note explaining why a catalog-managed record stays locked. A user who already moved focus elsewhere keeps it.
  const retryButtonRef = useRef<HTMLButtonElement | null>(null);
  const attachRetryButton = useCallback((node: HTMLButtonElement | null) => {
    const previous = retryButtonRef.current;

    retryButtonRef.current = node;

    if (node || !previous?.contains(document.activeElement)) {
      return;
    }

    // Detaching runs before the node leaves the document; hand focus on once the replacement has committed.
    queueMicrotask(() => {
      focusIfUnclaimed(sectionRef.current);

      if (document.activeElement === null || document.activeElement === document.body) {
        starterManagedNoteRef.current?.focus();
      }
    });
  }, []);

  const numberField = ({ key, labelKey }: { key: OptionalNumberKey; labelKey: string }, placeholderKey: string) => (
    <OptionalNumberField
      key={key}
      disabled={isSaving}
      error={errorFor(key)}
      label={t(labelKey)}
      placeholder={t(placeholderKey)}
      setValue={(value) => setField(key, value)}
      value={values[key]}
    />
  );

  return (
    <Stack ref={sectionRef} gap="3">
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
      {hasModelDraftConflict(saved, draft) ? <DraftConflictNotice /> : null}
      {starters === null ? (
        <Panel gap="2" p="2.5" tone="surface">
          {/* The status region stays mounted so a retry's "loading" is announced as a change; the alert mounts afresh
              for each failed attempt, so a repeated failure is announced again. */}
          <Box>
            <Text color="fg.muted" fontSize="xs" role="status">
              {startersStatus === 'error' ? null : t('models.externalFields.catalogLoading')}
            </Text>
            {startersStatus === 'error' ? (
              <Text color="fg.error" fontSize="xs" role="alert">
                {t('models.externalFields.catalogUnavailable')}
              </Text>
            ) : null}
          </Box>
          {showsCatalogRetry ? (
            // `aria-disabled` rather than `disabled`: a disabled button drops the focus it holds.
            <Button
              ref={attachRetryButton}
              alignSelf="start"
              aria-busy={isRetryingCatalog || undefined}
              aria-disabled={isRetryingCatalog || undefined}
              variant="outline"
              onClick={handleRetryCatalog}
            >
              {isRetryingCatalog ? <Spinner boxSize="3" /> : null}
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
        <Field error={errorFor('providerId')} helpText={t('models.providerIdHelp')} label={t('models.providerId')}>
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
        <Field
          error={errorFor('providerModelId')}
          helpText={t('models.providerModelIdHelp')}
          label={t('models.providerModelId')}
        >
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
            <Text ref={starterManagedNoteRef} color="fg.muted" fontSize="xs" role="note" tabIndex={-1}>
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
          <Panel
            aria-describedby={modesMessageId}
            aria-labelledby={modesHeadingId}
            data-invalid={modesError ? '' : undefined}
            gap="2"
            p="2.5"
            role="group"
            tone="surface"
          >
            <Text fontSize="xs" fontWeight="600" id={modesHeadingId} textTransform="uppercase">
              {t('models.externalFields.supportedModes')}
            </Text>
            <HStack gap="4" wrap="wrap">
              {MODE_FLAGS.map(({ key, labelKey }) => (
                <Checkbox.Root
                  key={key}
                  checked={values[key]}
                  disabled={isSaving}
                  invalid={modesError !== null}
                  size="sm"
                  onCheckedChange={(event) => setField(key, event.checked === true)}
                >
                  <Checkbox.HiddenInput />
                  <Checkbox.Control />
                  <Checkbox.Label fontSize="xs">{t(labelKey)}</Checkbox.Label>
                </Checkbox.Root>
              ))}
            </HStack>
            {modesError ? (
              <Text color="fg.error" fontSize="xs" id={modesMessageId} role="alert">
                {modesError}
              </Text>
            ) : (
              <Text color="fg.subtle" fontSize="xs" id={modesMessageId}>
                {t('models.externalFields.supportedModesHelp')}
              </Text>
            )}
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
