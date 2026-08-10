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
import { useZodForm } from '@platform/react/useZodForm';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, Field, Select } from '@platform/ui';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

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
  const form = useZodForm(modelEditSchema, {
    base: String(model.base),
    configPath: model.config_path ?? '',
    description: model.description ?? '',
    format: String(model.format),
    name: model.name,
    predictionType: (model.prediction_type ?? '') as ModelEditFormValues['predictionType'],
    sourceUrl: model.source_url ?? '',
    type: String(model.type),
    variant: model.variant ?? '',
  });

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
  // Reacts to base/type edits so the offered variants always match the
  // combination being saved; an unknown current value stays selectable.
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

  const handleSave = () =>
    form.handleSubmit(async (values) => {
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
        onSaved();
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        throw error;
      }
    });

  return (
    <Stack gap="3">
      <Field error={form.errors.name} label={t('common.name')}>
        <Input
          aria-invalid={form.errors.name ? true : undefined}
          size="sm"
          value={form.values.name}
          onChange={(event) => form.setValue('name', event.currentTarget.value)}
        />
      </Field>
      <Field error={form.errors.description} label={t('models.description')}>
        <Textarea
          rows={2}
          size="sm"
          value={form.values.description}
          onChange={(event) => form.setValue('description', event.currentTarget.value)}
        />
      </Field>
      <HStack align="start" gap="2">
        <Field error={form.errors.base} label={t('models.base')}>
          <Select
            aria-label={t('models.base')}
            collection={baseCollection}
            size="sm"
            value={[form.values.base]}
            onValueChange={({ value }) => {
              const base = value[0];

              if (base !== undefined) {
                form.setValue('base', base);
              }
            }}
          />
        </Field>
        <Field error={form.errors.type} label={t('models.type')}>
          <Select
            aria-label={t('models.type')}
            collection={MODEL_TYPE_COLLECTION}
            size="sm"
            value={[form.values.type]}
            onValueChange={({ value }) => {
              const type = value[0];

              if (type !== undefined) {
                form.setValue('type', type);
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
              size="sm"
              value={[form.values.variant]}
              onValueChange={({ value }) => {
                const variant = value[0];

                if (variant !== undefined) {
                  form.setValue('variant', variant);
                }
              }}
            />
          ) : (
            <Input
              size="sm"
              value={form.values.variant}
              onChange={(event) => form.setValue('variant', event.currentTarget.value)}
            />
          )}
        </Field>
        <Field error={form.errors.predictionType} label={t('models.predictionType')}>
          <Select
            aria-label={t('models.predictionType')}
            collection={predictionTypeCollection}
            size="sm"
            value={[form.values.predictionType]}
            onValueChange={({ value }) => {
              const predictionType = value[0];

              if (predictionType !== undefined) {
                form.setValue('predictionType', predictionType as ModelEditFormValues['predictionType']);
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
            size="sm"
            value={[form.values.format]}
            onValueChange={({ value }) => {
              const format = value[0];

              if (format !== undefined) {
                form.setValue('format', format);
              }
            }}
          />
        </Field>
        {hasConfigPath ? (
          <Field error={form.errors.configPath} helpText={t('models.configPathHelp')} label={t('models.configPath')}>
            <Input
              size="sm"
              value={form.values.configPath}
              onChange={(event) => form.setValue('configPath', event.currentTarget.value)}
            />
          </Field>
        ) : null}
      </HStack>
      <Field error={form.errors.sourceUrl} helpText={t('models.sourceUrlHelp')} label={t('models.sourceUrl')}>
        <Input
          aria-invalid={form.errors.sourceUrl ? true : undefined}
          placeholder="https://…"
          size="sm"
          value={form.values.sourceUrl}
          onChange={(event) => form.setValue('sourceUrl', event.currentTarget.value)}
        />
      </Field>
      {form.formError ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {form.formError}
        </Text>
      ) : null}
      <HStack gap="2" justify="flex-end">
        <Button disabled={form.isSubmitting} size="xs" variant="ghost" onClick={onCancel}>
          {t('common.cancel')}
        </Button>
        <Button loading={form.isSubmitting} size="xs" variant="solid" onClick={() => void handleSave()}>
          {t('users.saveChanges')}
        </Button>
      </HStack>
    </Stack>
  );
};
