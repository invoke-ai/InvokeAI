import type { AnyModelDefaultSettings, ModelConfig, ModelTaxonomyType } from '@workbench/models/types';

import { DataList, HStack, Icon, Menu, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, ConfirmDialog, MenuContent } from '@workbench/components/ui';
import { isConvertibleToDiffusers } from '@workbench/models/baseIdentity';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { formatBytes } from '@workbench/models/taxonomy';
import { useNotify } from '@workbench/useNotify';
import { areArraysEqual } from '@workbench/workbenchSelectors';
import { ArrowLeftIcon, MoreHorizontalIcon, PencilIcon, RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { memo, useCallback, useState } from 'react';
import { SiHuggingface } from 'react-icons/si';

import { DefaultSettingsSection, supportsDefaultSettings } from './DefaultSettingsSection';
import { MissingFileBadge, ModelBaseBadge, ModelFormatBadge } from './ModelBadges';
import { ModelEditForm } from './ModelEditForm';
import { ModelImageUpload } from './ModelImageUpload';
import { RelatedModelsSection } from './RelatedModelsSection';
import { MemoizedTriggerPhrasesEditor } from './TriggerPhrasesEditor';
import { useModelActions } from './useModelActions';

const TRIGGER_PHRASE_TYPES = new Set(['main', 'lora', 'embedding']);
const RELATED_MODEL_TYPES = new Set(['main', 'lora']);
const EMPTY_TRIGGER_PHRASES: readonly string[] = [];

type ModelDetailShellModel = Pick<ModelConfig, 'key' | 'type'>;
type DefaultSettingsModel = { default_settings?: AnyModelDefaultSettings | null; key: string; type: ModelTaxonomyType };
type TriggerPhrasesModel = Pick<ModelConfig, 'key' | 'trigger_phrases'>;
type ModelIdentityModel = Pick<
  ModelConfig,
  | 'base'
  | 'cover_image'
  | 'description'
  | 'file_size'
  | 'format'
  | 'hash'
  | 'key'
  | 'name'
  | 'path'
  | 'prediction_type'
  | 'source'
  | 'source_url'
  | 'type'
  | 'variant'
>;

const findModel = (models: readonly ModelConfig[], modelKey: string): ModelConfig | undefined =>
  models.find((candidate) => candidate.key === modelKey);

const selectModelShell = (models: readonly ModelConfig[], modelKey: string): ModelDetailShellModel | null => {
  const model = findModel(models, modelKey);

  return model ? { key: model.key, type: model.type } : null;
};

const selectModelIdentity = (models: readonly ModelConfig[], modelKey: string): ModelIdentityModel | null => {
  const model = findModel(models, modelKey);

  return model
    ? {
        base: model.base,
        cover_image: model.cover_image,
        description: model.description,
        file_size: model.file_size,
        format: model.format,
        hash: model.hash,
        key: model.key,
        name: model.name,
        path: model.path,
        prediction_type: model.prediction_type,
        source: model.source,
        source_url: model.source_url,
        type: model.type,
        variant: model.variant,
      }
    : null;
};

const selectDefaultSettingsModel = (models: readonly ModelConfig[], modelKey: string): DefaultSettingsModel | null => {
  const model = findModel(models, modelKey);

  return model ? { default_settings: model.default_settings, key: model.key, type: model.type } : null;
};

const selectTriggerPhrasesModel = (models: readonly ModelConfig[], modelKey: string): TriggerPhrasesModel | null => {
  const model = findModel(models, modelKey);

  return model ? { key: model.key, trigger_phrases: model.trigger_phrases } : null;
};

const areTriggerPhrasesModelsEqual = (left: TriggerPhrasesModel | null, right: TriggerPhrasesModel | null): boolean =>
  left?.key === right?.key && areArraysEqual(left?.trigger_phrases ?? [], right?.trigger_phrases ?? []);

/**
 * Full detail pane for one model: identity (view/edit), per-model default
 * settings, related models, trigger phrases, and lifecycle actions (convert,
 * re-identify, delete). Mount keyed by model key so per-model form state never
 * leaks between models. `density="panel"` tightens the side-panel drill-in.
 */
export const ModelDetail = ({
  density = 'full',
  modelKey,
  onBack,
  onDeleted,
}: {
  density?: 'panel' | 'full';
  modelKey: string;
  onBack?: () => void;
  onDeleted: () => void;
}) => {
  const model = useModelsSelector((snapshot) => selectModelShell(snapshot.models, modelKey));

  if (!model) {
    return (
      <Stack align="start" gap="2" p="1">
        {onBack ? <BackButton onBack={onBack} /> : null}
        <Text color="fg.subtle" fontSize="xs">
          This model is no longer in the library.
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap={density === 'panel' ? '3' : '4'} pb="4">
      {onBack ? <BackButton onBack={onBack} /> : null}

      <ModelIdentitySectionContainer density={density} modelKey={model.key} onDeleted={onDeleted} />

      {supportsDefaultSettings(model) ? (
        <>
          <Separator borderColor="border.subtle" />
          <DefaultSettingsSectionContainer modelKey={model.key} />
        </>
      ) : null}

      {RELATED_MODEL_TYPES.has(model.type) ? (
        <>
          <Separator borderColor="border.subtle" />
          <RelatedModelsSectionContainer modelKey={model.key} />
        </>
      ) : null}

      {TRIGGER_PHRASE_TYPES.has(model.type) ? (
        <>
          <Separator borderColor="border.subtle" />
          <TriggerPhrasesEditorContainer modelKey={model.key} />
        </>
      ) : null}
    </Stack>
  );
};

interface ModelIdentitySectionProps {
  density: 'panel' | 'full';
  isMissing: boolean;
  model: ModelIdentityModel;
  onDeleted: () => void;
}

const ModelIdentitySectionContainer = memo(function ModelIdentitySectionContainer({
  density,
  modelKey,
  onDeleted,
}: {
  density: 'panel' | 'full';
  modelKey: string;
  onDeleted: () => void;
}) {
  const model = useModelsSelector((snapshot) => selectModelIdentity(snapshot.models, modelKey));
  const isMissing = useModelsSelector((snapshot) => snapshot.missingModelKeys.has(modelKey));

  if (!model) {
    return null;
  }

  return <ModelIdentitySection density={density} isMissing={isMissing} model={model} onDeleted={onDeleted} />;
});

const ModelIdentitySection = memo(function ModelIdentitySection({
  density,
  isMissing,
  model,
  onDeleted,
}: ModelIdentitySectionProps) {
  const notify = useNotify();
  const [editingModelKey, setEditingModelKey] = useState<string | null>(null);
  const isEditing = editingModelKey === model.key;

  return (
    <>
      <HStack align="start" gap="3">
        <ModelImageUpload
          key={model.key}
          model={model}
          onError={(message) => notify.error('Model image', message)}
          onUpdated={() => notify.success('Model image updated')}
        />
        <Stack flex="1" gap="1" minW="0">
          <Text fontSize="sm" fontWeight="700" lineClamp={2}>
            {model.name}
          </Text>
          <HStack gap="1" minW="0" wrap="wrap">
            <ModelBaseBadge base={model.base} />
            <ModelFormatBadge format={model.format} />
          </HStack>
          {isMissing ? (
            <HStack gap="1.5">
              <MissingFileBadge />
              <Text color="fg.subtle" fontSize="2xs">
                The file for this model was not found on disk.
              </Text>
            </HStack>
          ) : null}
          {model.description ? (
            <Text color="fg.muted" fontSize="2xs" lineClamp={3}>
              {model.description}
            </Text>
          ) : null}
        </Stack>
        <ModelDetailActions
          isEditing={isEditing}
          model={model}
          onDeleted={onDeleted}
          onToggleEditing={() => setEditingModelKey((key) => (key === model.key ? null : model.key))}
        />
      </HStack>

      {isEditing ? (
        <ModelEditForm
          model={model}
          onCancel={() => setEditingModelKey(null)}
          onSaved={() => {
            setEditingModelKey(null);
            notify.success('Model updated', model.name);
          }}
        />
      ) : (
        <ModelAttributes density={density} model={model} />
      )}
    </>
  );
});

const ModelDetailActions = ({
  isEditing,
  model,
  onDeleted,
  onToggleEditing,
}: {
  isEditing: boolean;
  model: ModelIdentityModel;
  onDeleted: () => void;
  onToggleEditing: () => void;
}) => {
  const { convert, reidentify, remove } = useModelActions();
  const [pendingAction, setPendingAction] = useState<'delete' | 'convert' | null>(null);
  const [isActionBusy, setIsActionBusy] = useState(false);

  const handleReidentify = async () => {
    setIsActionBusy(true);

    try {
      await reidentify(model);
    } finally {
      setIsActionBusy(false);
    }
  };

  return (
    <HStack flexShrink={0} gap="1" wrap="wrap">
      {isConvertibleToDiffusers(model) ? (
        <Button size="xs" variant="outline" onClick={() => setPendingAction('convert')}>
          <Icon as={SiHuggingface} boxSize="3" />
          Convert to Diffusers
        </Button>
      ) : null}
      <Button size="xs" variant={isEditing ? 'solid' : 'outline'} onClick={onToggleEditing}>
        <Icon as={PencilIcon} boxSize="3" />
        {isEditing ? 'Editing' : 'Edit'}
      </Button>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Menu.Trigger asChild>
          <IconButton aria-label="Model actions" loading={isActionBusy} size="xs" variant="ghost">
            <Icon as={MoreHorizontalIcon} boxSize="4" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="12rem">
              <Menu.Item value="reidentify" onClick={() => void handleReidentify()}>
                <Icon as={RefreshCcwIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">Re-identify model</Menu.ItemText>
              </Menu.Item>
              <Menu.Separator />
              <Menu.Item color="fg.error" value="delete" onClick={() => setPendingAction('delete')}>
                <Icon as={Trash2Icon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">Delete model</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ConfirmDialog
        body={`Delete “${model.name}”? The database record is removed, and the model files are deleted if they live inside the InvokeAI models directory.`}
        confirmLabel="Delete Model"
        isOpen={pendingAction === 'delete'}
        title="Delete model"
        onClose={() => setPendingAction(null)}
        onConfirm={async () => {
          await remove(model);
          onDeleted();
        }}
      />
      {/* Destructive styling: the original checkpoint file is replaced. */}
      <ConfirmDialog
        body={`Convert “${model.name}” to the diffusers format in place? The original checkpoint file is replaced by a diffusers folder.`}
        confirmLabel="Convert"
        isOpen={pendingAction === 'convert'}
        title="Convert to diffusers"
        onClose={() => setPendingAction(null)}
        onConfirm={() => convert(model)}
      />
    </HStack>
  );
};

const DefaultSettingsSectionContainer = memo(function DefaultSettingsSectionContainer({
  modelKey,
}: {
  modelKey: string;
}) {
  const notify = useNotify();
  const model = useModelsSelector((snapshot) => selectDefaultSettingsModel(snapshot.models, modelKey));

  if (!model || !supportsDefaultSettings(model)) {
    return null;
  }

  return (
    <DefaultSettingsSection
      model={model}
      onError={(message) => notify.error('Default settings', message)}
      onSaved={() => notify.success('Default settings saved')}
    />
  );
});

const TriggerPhrasesEditorContainer = memo(function TriggerPhrasesEditorContainer({ modelKey }: { modelKey: string }) {
  const notify = useNotify();
  const handleError = useCallback((message: string) => notify.error('Trigger phrases', message), [notify]);
  const model = useModelsSelector(
    (snapshot) => selectTriggerPhrasesModel(snapshot.models, modelKey),
    areTriggerPhrasesModelsEqual
  );

  if (!model) {
    return null;
  }

  return (
    <MemoizedTriggerPhrasesEditor
      modelKey={model.key}
      phrases={model.trigger_phrases ?? EMPTY_TRIGGER_PHRASES}
      onError={handleError}
    />
  );
});

const RelatedModelsSectionContainer = memo(function RelatedModelsSectionContainer({ modelKey }: { modelKey: string }) {
  const notify = useNotify();
  const model = useModelsSelector(
    (snapshot) => {
      const candidate = snapshot.models.find((model) => model.key === modelKey);

      return candidate ? { base: candidate.base, key: candidate.key } : null;
    },
    (left, right) => left?.key === right?.key && left?.base === right?.base
  );
  const handleSectionError = useCallback((message: string) => notify.error('Model manager', message), [notify]);

  if (!model) {
    return null;
  }

  return <RelatedModelsSection model={model} onError={handleSectionError} />;
});

const BackButton = ({ onBack }: { onBack: () => void }) => (
  <Button alignSelf="start" size="2xs" variant="ghost" onClick={onBack}>
    <Icon as={ArrowLeftIcon} boxSize="3" />
    All models
  </Button>
);

const isAbsolutePath = (path: string): boolean => path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path);

const ModelAttributes = ({ density, model }: { density: 'panel' | 'full'; model: ModelIdentityModel }) => {
  const modelsDir = useModelsSelector((snapshot) => snapshot.modelsDir);
  // Managed models store paths relative to the models directory; show the
  // resolved absolute path so it can be found on disk.
  const fullPath =
    isAbsolutePath(model.path) || !modelsDir ? model.path : `${modelsDir.replace(/\/+$/, '')}/${model.path}`;

  const attributes: { label: string; value: string }[] = [
    { label: 'File Size', value: formatBytes(model.file_size) },
    { label: 'Variant', value: model.variant ?? '—' },
    { label: 'Prediction Type', value: model.prediction_type ?? '—' },
    { label: 'Hash', value: model.hash },
    { label: 'Path', value: fullPath },
    { label: 'Source', value: model.source },
  ];

  return (
    <DataList.Root gap="2.5" orientation={density === 'panel' ? 'vertical' : 'horizontal'} size="sm" variant="subtle">
      {attributes.map((attribute) => (
        <DataList.Item key={attribute.label}>
          <DataList.ItemLabel color="fg.subtle" fontSize="2xs" minW="8rem" textTransform="uppercase">
            {attribute.label}
          </DataList.ItemLabel>
          <DataList.ItemValue fontSize="2xs" overflowWrap="anywhere">
            {attribute.value}
          </DataList.ItemValue>
        </DataList.Item>
      ))}
    </DataList.Root>
  );
};
