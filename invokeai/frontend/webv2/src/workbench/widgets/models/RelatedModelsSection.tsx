import { Badge, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { Link2OffIcon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { ModelSelect } from '@workbench/components/ModelSelect';
import { IconButton } from '@workbench/components/ui/Button';
import { FieldLabel } from '@workbench/components/ui/Field';
import { Tooltip } from '@workbench/components/ui/Tooltip';
import { addModelRelationship, getRelatedModelKeys, removeModelRelationship } from '@workbench/models/api';
import { useModelsSnapshot } from '@workbench/models/modelsStore';
import { getModelBaseColorPalette, getModelBaseLabel, getModelTypeLabel } from '@workbench/models/taxonomy';
import type { ModelConfig, ModelTaxonomyType } from '@workbench/models/types';
import { SourceListItem } from './SourceListItem';

/** Types offered when linking related models, grouped in one picker. */
const LINKABLE_TYPES: ModelTaxonomyType[] = [
  'main',
  'lora',
  'embedding',
  'vae',
  'controlnet',
  't2i_adapter',
  'ip_adapter',
];

/**
 * Bidirectional "related models" links — e.g. attach the LoRAs and VAE that
 * pair well with a checkpoint so other surfaces can suggest them together.
 * One grouped, compatibility-filtered picker searches every linkable type at
 * once; linked models render as rows with an unlink action.
 */
export const RelatedModelsSection = ({
  model,
  onError,
}: {
  model: ModelConfig;
  onError: (message: string) => void;
}) => {
  const { models } = useModelsSnapshot();
  const [relatedKeys, setRelatedKeys] = useState<string[] | null>(null);
  const [isMutating, setIsMutating] = useState(false);

  useEffect(() => {
    let isStale = false;

    getRelatedModelKeys(model.key)
      .then((keys) => {
        if (!isStale) {
          setRelatedKeys(keys);
        }
      })
      .catch((error: unknown) => {
        if (!isStale) {
          setRelatedKeys([]);
          onError(error instanceof Error ? error.message : 'Failed to load related models.');
        }
      });

    return () => {
      isStale = true;
    };
  }, [model.key, onError]);

  const relatedModels = useMemo(() => {
    const byKey = new Map(models.map((candidate) => [candidate.key, candidate]));

    return (relatedKeys ?? []).map((key) => ({ key, model: byKey.get(key) ?? null }));
  }, [models, relatedKeys]);

  const excludeKeys = useMemo(() => new Set([model.key, ...(relatedKeys ?? [])]), [model.key, relatedKeys]);

  const handleAdd = async (target: ModelConfig | null) => {
    if (!target || isMutating) {
      return;
    }

    setIsMutating(true);

    try {
      await addModelRelationship(model.key, target.key);
      setRelatedKeys((current) => [...(current ?? []), target.key]);
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Failed to link models.');
    } finally {
      setIsMutating(false);
    }
  };

  const handleRemove = async (key: string) => {
    setIsMutating(true);

    try {
      await removeModelRelationship(model.key, key);
      setRelatedKeys((current) => (current ?? []).filter((existing) => existing !== key));
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Failed to unlink models.');
    } finally {
      setIsMutating(false);
    }
  };

  return (
    <Stack gap="2">
      <Stack gap="0.5">
        <FieldLabel>Related Models</FieldLabel>
        <Text color="fg.subtle" fontSize="2xs">
          Link the models that pair well with this one — they are suggested together elsewhere in the app.
        </Text>
      </Stack>
      <ModelSelect
        excludeKeys={excludeKeys}
        filter={(candidate) => candidate.base === model.base || candidate.base === 'any' || model.base === 'any'}
        modelTypes={LINKABLE_TYPES}
        placeholder="Search compatible models to link…"
        size="sm"
        value={null}
        onChange={(target) => {
          void handleAdd(target);
        }}
      />
      {relatedKeys === null ? (
        <Spinner color="fg.subtle" size="xs" />
      ) : relatedModels.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          No related models linked yet.
        </Text>
      ) : (
        <Stack gap="1.5">
          {relatedModels.map(({ key, model: relatedModel }) => (
            <SourceListItem
              key={key}
              badges={
                relatedModel ? (
                  <>
                    <Badge
                      colorPalette={getModelBaseColorPalette(relatedModel.base)}
                      flexShrink={0}
                      fontSize="2xs"
                      size="sm"
                      variant="surface"
                    >
                      {getModelBaseLabel(relatedModel.base)}
                    </Badge>
                    <Badge colorPalette="gray" flexShrink={0} fontSize="2xs" size="sm" variant="outline">
                      {getModelTypeLabel(relatedModel.type)}
                    </Badge>
                  </>
                ) : undefined
              }
              title={relatedModel?.name ?? key}
              trailing={
                <Tooltip content="Unlink">
                  <IconButton
                    aria-label={`Unlink ${relatedModel?.name ?? key}`}
                    disabled={isMutating}
                    size="2xs"
                    variant="ghost"
                    onClick={() => {
                      void handleRemove(key);
                    }}
                  >
                    <Icon as={Link2OffIcon} boxSize="3" />
                  </IconButton>
                </Tooltip>
              }
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
