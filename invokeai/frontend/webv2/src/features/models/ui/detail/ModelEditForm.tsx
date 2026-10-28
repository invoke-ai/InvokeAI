/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig, PredictionType } from '@features/models/core/types';

import { createListCollection, HStack, Input, Stack, Text, Textarea } from '@chakra-ui/react';
import { getModelBaseLabel, KNOWN_MODEL_BASES } from '@features/models/core/baseIdentity';
import { modelEditSchema, type ModelEditFormValues } from '@features/models/core/schemas';
import {
  EDITABLE_MODEL_FORMATS,
  getModelFormatLabel,
  getModelTypeLabel,
  getModelVariantLabel,
  getVariantOptionsFor,
  MODEL_CATEGORIES,
} from '@features/models/core/taxonomy';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import {
  applyModelDraftFields,
  beginModelDraftSave,
  clearModelIdentitySaveError,
  failModelDraftSave,
  finishModelDraftSave,
  hasDraftFields,
  hasModelDraftConflict,
  recordModelDraftFields,
  toModelIdentityValues,
  useModelDraft,
  type ModelDraft,
} from '@features/models/ui/modelDraftsStore';
import { useZodForm } from '@platform/react/useZodForm';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { shallowEqual } from '@platform/state/selectors';
import { Button, Field, Select } from '@platform/ui';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { DraftConflictNotice, UnsavedChangesLabel } from './ModelDraftNotices';

// `external` is the hosted-provider sentinel; assigning it to a local model
// would misroute it across the app, so the edit form never offers it.
const ASSIGNABLE_BASES: readonly string[] = KNOWN_MODEL_BASES.filter((base) => base !== 'external');

const MODEL_TYPE_COLLECTION = createListCollection({
  items: MODEL_CATEGORIES.map((category) => ({ label: getModelTypeLabel(category.type), value: category.type })),
});

/** Zod-validated editor for a model's identity fields. */
type ModelEditTarget = Pick<
  ModelConfig,
  | 'base'
  | 'config_path'
  | 'description'
  | 'format'
  | 'key'
  | 'name'
  | 'prediction_type'
  | 'source_url'
  | 'type'
  | 'variant'
>;

const selectIdentityDraft = (draft: ModelDraft | undefined) => draft?.identity;

/**
 * Edits write through to the model's retained draft, so the form reopens as it was left until saved or cancelled. The
 * draft also carries a save in flight and its failure, which outlive this instance when the user navigates away.
 */
export const ModelEditForm = ({
  model,
  onCancel,
  onSaved,
}: {
  model: ModelEditTarget;
  onCancel: () => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  // `config_path` only exists on checkpoint-style config classes; its absence
  // (not emptiness) hides the field, since the PATCH would silently drop it.
  const hasConfigPath = model.config_path !== undefined;
  const serverValues = useMemo(() => toModelIdentityValues(model), [model]);
  const draft = useModelDraft(model.key, selectIdentityDraft);
  const form = useZodForm(modelEditSchema, applyModelDraftFields(serverValues, draft));
  const isSaving = draft?.submitted !== undefined;
  const formError = form.formError ?? draft?.error ?? null;
  // A server change rebases the form: edited fields keep the user's value, untouched fields take the new record.
  const [rebasedFrom, setRebasedFrom] = useState(serverValues);

  if (!shallowEqual(rebasedFrom, serverValues)) {
    setRebasedFrom(serverValues);
    form.setValues(applyModelDraftFields(serverValues, draft));
  }

  const setField = <Key extends keyof ModelEditFormValues>(key: Key, value: ModelEditFormValues[Key]) => {
    form.setValue(key, value);
    recordModelDraftFields(model.key, 'identity', { ...form.values, [key]: value }, serverValues);
  };

  const baseCollection = useMemo(() => {
    const bases: readonly string[] = ASSIGNABLE_BASES.includes(String(model.base))
      ? ASSIGNABLE_BASES
      : [String(model.base), ...ASSIGNABLE_BASES];

    return createListCollection({
      items: bases.map((base) => ({ label: getModelBaseLabel(base), value: base })),
    });
  }, [model.base]);
  const predictionTypeCollection = useMemo(
    () =>
      createListCollection({
        items: [
          { label: t('common.none'), value: '' },
          { label: 'epsilon', value: 'epsilon' },
          { label: 'v_prediction', value: 'v_prediction' },
          { label: 'sample', value: 'sample' },
        ],
      }),
    [t]
  );
  const formatCollection = useMemo(() => {
    const formats: readonly string[] = EDITABLE_MODEL_FORMATS.includes(String(model.format))
      ? EDITABLE_MODEL_FORMATS
      : [String(model.format), ...EDITABLE_MODEL_FORMATS];

    return createListCollection({
      items: formats.map((format) => ({ label: getModelFormatLabel(format), value: format })),
    });
  }, [model.format]);
  // Update variant choices with edited base/type while retaining unknown current values.
  const { variantCollection, variantOptions } = useMemo(() => {
    const options = getVariantOptionsFor(form.values.base, form.values.type);
    const withCurrent =
      form.values.variant !== '' && !options.includes(form.values.variant)
        ? [form.values.variant, ...options]
        : options;

    return {
      variantCollection: createListCollection({
        items: [
          { label: t('common.none'), value: '' },
          ...withCurrent.map((variant) => ({ label: getModelVariantLabel(variant), value: variant })),
        ],
      }),
      variantOptions: options,
    };
  }, [form.values.base, form.values.type, form.values.variant, t]);

  const handleSave = () => {
    clearModelIdentitySaveError(model.key);
    // The raw values, not the parsed ones: the draft's fields are raw, and a later edit is told apart by comparison.
    const submitted = form.values;

    return form.handleSubmit(async (values) => {
      if (!beginModelDraftSave(model.key, 'identity', submitted)) {
        return;
      }

      const owner = captureAccountScope();

      try {
        const updated = await updateModel(
          model.key,
          {
            base: values.base,
            description: values.description || null,
            format: values.format,
            name: values.name,
            prediction_type: values.predictionType === '' ? null : (values.predictionType as PredictionType),
            source_url: values.sourceUrl === '' ? null : values.sourceUrl,
            type: values.type,
            variant: values.variant === '' ? null : values.variant,
            ...(hasConfigPath ? { config_path: values.configPath === '' ? null : values.configPath } : {}),
          },
          owner.signal
        );

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        finishModelDraftSave(model.key, 'identity', toModelIdentityValues(updated));
        onSaved();
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        // Held by the draft rather than this instance, so whichever form is mounted when it lands shows it.
        failModelDraftSave(
          model.key,
          'identity',
          serverValues,
          error instanceof Error ? error.message : t('common.somethingWentWrong')
        );
      }
    });
  };

  return (
    <Stack gap="3">
      <Field error={form.errors.name} label={t('common.name')}>
        <Input
          aria-invalid={form.errors.name ? true : undefined}
          size="lg"
          value={form.values.name}
          onChange={(event) => setField('name', event.currentTarget.value)}
        />
      </Field>
      <Field error={form.errors.description} label={t('models.description')}>
        <Textarea
          rows={2}
          size="lg"
          value={form.values.description}
          onChange={(event) => setField('description', event.currentTarget.value)}
        />
      </Field>
      <HStack align="start" gap="2">
        <Field error={form.errors.base} label={t('models.base')}>
          <Select
            aria-label={t('models.base')}
            collection={baseCollection}
            size="lg"
            value={[form.values.base]}
            onValueChange={({ value }) => {
              const base = value[0];

              if (base !== undefined) {
                setField('base', base);
              }
            }}
          />
        </Field>
        <Field error={form.errors.type} label={t('models.type')}>
          <Select
            aria-label={t('models.type')}
            collection={MODEL_TYPE_COLLECTION}
            size="lg"
            value={[form.values.type]}
            onValueChange={({ value }) => {
              const type = value[0];

              if (type !== undefined) {
                setField('type', type);
              }
            }}
          />
        </Field>
      </HStack>
      <HStack align="start" gap="2">
        <Field error={form.errors.variant} helpText={t('models.variantHelp')} label={t('models.variant')}>
          {variantOptions.length > 0 ? (
            <Select
              aria-label={t('models.variant')}
              collection={variantCollection}
              size="lg"
              value={[form.values.variant]}
              onValueChange={({ value }) => {
                const variant = value[0];

                if (variant !== undefined) {
                  setField('variant', variant);
                }
              }}
            />
          ) : (
            <Input
              size="lg"
              value={form.values.variant}
              onChange={(event) => setField('variant', event.currentTarget.value)}
            />
          )}
        </Field>
        <Field error={form.errors.predictionType} label={t('models.predictionType')}>
          <Select
            aria-label={t('models.predictionType')}
            collection={predictionTypeCollection}
            size="lg"
            value={[form.values.predictionType]}
            onValueChange={({ value }) => {
              const predictionType = value[0];

              if (predictionType !== undefined) {
                setField('predictionType', predictionType as ModelEditFormValues['predictionType']);
              }
            }}
          />
        </Field>
      </HStack>
      <HStack align="start" gap="2">
        <Field error={form.errors.format} helpText={t('models.formatHelp')} label={t('models.format')}>
          <Select
            aria-label={t('models.format')}
            collection={formatCollection}
            size="lg"
            value={[form.values.format]}
            onValueChange={({ value }) => {
              const format = value[0];

              if (format !== undefined) {
                setField('format', format);
              }
            }}
          />
        </Field>
        {hasConfigPath ? (
          <Field error={form.errors.configPath} helpText={t('models.configPathHelp')} label={t('models.configPath')}>
            <Input
              size="lg"
              value={form.values.configPath}
              onChange={(event) => setField('configPath', event.currentTarget.value)}
            />
          </Field>
        ) : null}
      </HStack>
      <Field error={form.errors.sourceUrl} helpText={t('models.sourceUrlHelp')} label={t('models.sourceUrl')}>
        <Input
          aria-invalid={form.errors.sourceUrl ? true : undefined}
          placeholder="https://…"
          size="lg"
          value={form.values.sourceUrl}
          onChange={(event) => setField('sourceUrl', event.currentTarget.value)}
        />
      </Field>
      {formError ? (
        <Text color="fg.error" fontSize="xs" role="alert">
          {formError}
        </Text>
      ) : null}
      {hasModelDraftConflict(serverValues, draft) ? <DraftConflictNotice /> : null}
      <HStack gap="2" justify="flex-end">
        {hasDraftFields(draft) ? <UnsavedChangesLabel /> : null}
        <Button disabled={isSaving} variant="ghost" onClick={onCancel}>
          {t('common.cancel')}
        </Button>
        <Button loading={isSaving} variant="solid" onClick={() => void handleSave()}>
          {t('users.saveChanges')}
        </Button>
      </HStack>
    </Stack>
  );
};
