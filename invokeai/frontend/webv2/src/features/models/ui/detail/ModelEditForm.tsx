/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig, PredictionType } from '@features/models/core/types';

import { createListCollection, HStack, Input, Stack, Text, Textarea } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models/core/baseIdentity';
import { modelEditSchema, type ModelEditFormValues } from '@features/models/core/schemas';
import { getModelTypeLabel, MODEL_CATEGORIES } from '@features/models/core/taxonomy';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { useZodForm } from '@platform/react/useZodForm';
import { Button, Field, Select } from '@platform/ui';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const KNOWN_BASES = [
  'sd-1',
  'sd-2',
  'sd-3',
  'sdxl',
  'sdxl-refiner',
  'flux',
  'flux2',
  'cogview4',
  'qwen-image',
  'z-image',
  'anima',
  'any',
  'unknown',
];

const MODEL_TYPE_COLLECTION = createListCollection({
  items: MODEL_CATEGORIES.map((category) => ({ label: getModelTypeLabel(category.type), value: category.type })),
});

/** Zod-validated editor for a model's identity fields. */
type ModelEditTarget = Pick<
  ModelConfig,
  'base' | 'description' | 'key' | 'name' | 'prediction_type' | 'source_url' | 'type' | 'variant'
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
  const form = useZodForm(modelEditSchema, {
    base: String(model.base),
    description: model.description ?? '',
    name: model.name,
    predictionType: (model.prediction_type ?? '') as ModelEditFormValues['predictionType'],
    sourceUrl: model.source_url ?? '',
    type: String(model.type),
    variant: model.variant ?? '',
  });

  const baseCollection = useMemo(() => {
    const bases = KNOWN_BASES.includes(String(model.base)) ? KNOWN_BASES : [String(model.base), ...KNOWN_BASES];

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

  const handleSave = () =>
    form.handleSubmit(async (values) => {
      const updated = await updateModel(model.key, {
        base: values.base,
        description: values.description || null,
        name: values.name,
        prediction_type: values.predictionType === '' ? null : (values.predictionType as PredictionType),
        source_url: values.sourceUrl === '' ? null : values.sourceUrl,
        type: values.type,
        variant: values.variant === '' ? null : values.variant,
      });

      replaceModelInStore(updated);
      onSaved();
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
          <Input
            size="sm"
            value={form.values.variant}
            onChange={(event) => form.setValue('variant', event.currentTarget.value)}
          />
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
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
