import type { ModelConfig } from '@workbench/models/types';

import { DataList, HStack, Icon, Menu, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, ConfirmDialog, MenuContent } from '@workbench/components/ui';
import { useModelsSnapshot } from '@workbench/models/modelsStore';
import { formatBytes, isConvertibleToDiffusers } from '@workbench/models/taxonomy';
import { useNotify } from '@workbench/useNotify';
import { ArrowLeftIcon, MoreHorizontalIcon, PencilIcon, RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { SiHuggingface } from 'react-icons/si';

import { DefaultSettingsSection, supportsDefaultSettings } from './DefaultSettingsSection';
import { MissingFileBadge, ModelBadgeRow } from './ModelBadges';
import { ModelEditForm } from './ModelEditForm';
import { ModelImageUpload } from './ModelImageUpload';
import { RelatedModelsSection } from './RelatedModelsSection';
import { TriggerPhrasesEditor } from './TriggerPhrasesEditor';
import { useModelActions } from './useModelActions';

const TRIGGER_PHRASE_TYPES = new Set(['main', 'lora', 'embedding']);
const RELATED_MODEL_TYPES = new Set(['main', 'lora']);

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
  const notify = useNotify();
  const { missingModelKeys, models } = useModelsSnapshot();
  const { convert, reidentify, remove } = useModelActions();
  const model = models.find((candidate) => candidate.key === modelKey);
  const [isEditing, setIsEditing] = useState(false);
  const [pendingAction, setPendingAction] = useState<'delete' | 'convert' | null>(null);
  const [isActionBusy, setIsActionBusy] = useState(false);

  const handleSectionError = useCallback(
    (message: string) => {
      notify.error('Model manager', message);
    },
    [notify]
  );

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

  const handleReidentify = async () => {
    setIsActionBusy(true);

    try {
      await reidentify(model);
    } finally {
      setIsActionBusy(false);
    }
  };

  return (
    <Stack gap={density === 'panel' ? '3' : '4'} pb="4">
      {onBack ? <BackButton onBack={onBack} /> : null}

      <HStack align="start" gap="3">
        <ModelImageUpload
          model={model}
          onError={(message) => notify.error('Model image', message)}
          onUpdated={() => notify.success('Model image updated')}
        />
        <Stack flex="1" gap="1" minW="0">
          <Text fontSize="sm" fontWeight="700" lineClamp={2}>
            {model.name}
          </Text>
          <ModelBadgeRow model={model} />
          {missingModelKeys.has(model.key) ? (
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
        <HStack flexShrink={0} gap="1" wrap="wrap">
          {isConvertibleToDiffusers(model) ? (
            <Button size="xs" variant="outline" onClick={() => setPendingAction('convert')}>
              <Icon as={SiHuggingface} boxSize="3" />
              Convert to Diffusers
            </Button>
          ) : null}
          <Button
            size="xs"
            variant={isEditing ? 'solid' : 'outline'}
            onClick={() => setIsEditing((editing) => !editing)}
          >
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
        </HStack>
      </HStack>

      {isEditing ? (
        <ModelEditForm
          model={model}
          onCancel={() => setIsEditing(false)}
          onSaved={() => {
            setIsEditing(false);
            notify.success('Model updated', model.name);
          }}
        />
      ) : (
        <ModelAttributes density={density} model={model} />
      )}

      {supportsDefaultSettings(model) ? (
        <>
          <Separator borderColor="border.subtle" />
          <DefaultSettingsSection
            model={model}
            onError={(message) => notify.error('Default settings', message)}
            onSaved={() => notify.success('Default settings saved', model.name)}
          />
        </>
      ) : null}

      {RELATED_MODEL_TYPES.has(model.type) ? (
        <>
          <Separator borderColor="border.subtle" />
          <RelatedModelsSection model={model} onError={handleSectionError} />
        </>
      ) : null}

      {TRIGGER_PHRASE_TYPES.has(model.type) ? (
        <>
          <Separator borderColor="border.subtle" />
          <TriggerPhrasesEditor model={model} onError={(message) => notify.error('Trigger phrases', message)} />
        </>
      ) : null}

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
    </Stack>
  );
};

const BackButton = ({ onBack }: { onBack: () => void }) => (
  <Button alignSelf="start" size="2xs" variant="ghost" onClick={onBack}>
    <Icon as={ArrowLeftIcon} boxSize="3" />
    All models
  </Button>
);

const isAbsolutePath = (path: string): boolean => path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path);

const ModelAttributes = ({ density, model }: { density: 'panel' | 'full'; model: ModelConfig }) => {
  const { modelsDir } = useModelsSnapshot();
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
