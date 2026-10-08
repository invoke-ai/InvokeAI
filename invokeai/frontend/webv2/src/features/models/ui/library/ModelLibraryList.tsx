/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@features/models/core/types';
import type { ListRowProps } from '@platform/ui/list/List';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';

import { Flex, HStack, Icon, Image } from '@chakra-ui/react';
import { filterModels, groupModelsByType, type ModelLibraryFilters } from '@features/models/core/library';
import { getModelImageUrl } from '@features/models/data/api';
import { useModelsSelector } from '@features/models/data/modelsStore';
import { MissingFileBadge, ModelBaseBadge, ModelFormatBadge } from '@features/models/ui/detail/ModelBadges';
import { getLibraryScrollOffset, openModelManagerTab, saveLibraryScrollOffset } from '@features/models/ui/uiStore';
import { formatBytes } from '@platform/i18n/languages';
import { Button } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromSections } from '@platform/ui/list/listRows';
import { ArrowRightIcon, BoxIcon, CircleAlert } from 'lucide-react';
import { memo, useCallback, useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ModelRowContextMenu, type ModelContextMenuTarget } from './ModelRowContextMenu';

const getModelKey = (model: ModelConfig): string => model.key;

export const ModelLibraryList = ({
  activeModelKey,
  filters,
  instanceId,
  onActivate,
  onToggleSelected,
  selectedKeys,
}: {
  activeModelKey: string | null;
  filters: ModelLibraryFilters;
  /** Identifies this list's scroll-offset slot ('manager'). */
  instanceId: string;
  onActivate: (modelKey: string) => void;
  onToggleSelected: (modelKey: string) => void;
  selectedKeys: ReadonlySet<string>;
}) => {
  const { t } = useTranslation();
  const { coverImageVersions, error, missingModelKeys, models, status } = useModelsSelector(
    (snapshot) => ({
      coverImageVersions: snapshot.coverImageVersions,
      error: snapshot.error,
      missingModelKeys: snapshot.missingModelKeys,
      models: snapshot.models,
      status: snapshot.status,
    }),
    (left, right) =>
      left.coverImageVersions === right.coverImageVersions &&
      left.error === right.error &&
      left.missingModelKeys === right.missingModelKeys &&
      left.models === right.models &&
      left.status === right.status
  );
  const deferredFilters = useDeferredValue(filters);
  const [contextMenuTarget, setContextMenuTarget] = useState<ModelContextMenuTarget | null>(null);
  const openAddModels = () => {
    openModelManagerTab('add');
  };
  const handleContextMenu = useCallback(
    (modelKey: string, anchor: ListContextMenuAnchor) => setContextMenuTarget({ ...anchor, modelKey }),
    []
  );
  const handleCloseContextMenu = useCallback(() => {
    contextMenuTarget?.restoreFocus();
    setContextMenuTarget(null);
  }, [contextMenuTarget]);

  const rows = useMemo(
    () =>
      listRowsFromSections(
        groupModelsByType(filterModels(models, deferredFilters, missingModelKeys)).map((group) => ({
          items: group.models,
          key: group.type,
          label: group.label,
        })),
        getModelKey
      ),
    [deferredFilters, missingModelKeys, models]
  );

  // Persist the offset so switching tabs/regions and coming back lands here.
  const initialScrollOffset = useMemo(() => getLibraryScrollOffset(instanceId), [instanceId]);
  const persistScrollOffset = useCallback(
    (offset: number) => saveLibraryScrollOffset(instanceId, offset),
    [instanceId]
  );

  const renderItem = (model: ModelConfig, rowProps: ListRowProps) => (
    <ModelRow
      {...rowProps}
      base={model.base}
      coverImage={model.cover_image}
      fileSize={model.file_size}
      format={model.format}
      imageVersion={coverImageVersions[model.key]}
      isMenuOpen={contextMenuTarget?.modelKey === model.key}
      isMissing={missingModelKeys.has(model.key)}
      isSelected={selectedKeys.has(model.key)}
      modelKey={model.key}
      name={model.name}
      onActivate={onActivate}
      onContextMenu={handleContextMenu}
      onToggleSelected={onToggleSelected}
    />
  );

  return (
    <>
      <List
        activeKey={activeModelKey}
        density="comfortable"
        emptyState={
          <EmptyState
            title={models.length === 0 ? t('models.noneInstalled') : t('models.noneMatchFilters')}
            description={
              models.length === 0 ? t('models.noneInstalledDescription') : t('models.noneMatchFiltersDescription')
            }
            icon={<Icon as={CircleAlert} />}
          >
            <Button onClick={openAddModels} size="lg">
              {t('models.addModels')}
              <Icon as={ArrowRightIcon} />
            </Button>
          </EmptyState>
        }
        errorState={
          <EmptyState title={t('models.couldNotLoad')} description={error} icon={<Icon as={CircleAlert} />} danger />
        }
        initialScrollOffset={initialScrollOffset}
        label={t('models.library')}
        renderItem={renderItem}
        rows={rows}
        status={status === 'error' ? 'error' : status === 'loaded' ? 'ready' : 'loading'}
        onScrollOffsetPersist={persistScrollOffset}
      />
      <ModelRowContextMenu target={contextMenuTarget} onClose={handleCloseContextMenu} />
    </>
  );
};

interface ModelRowProps extends ListRowProps {
  base: Parameters<typeof ModelBaseBadge>[0]['base'];
  coverImage?: string | null;
  fileSize: number;
  format: Parameters<typeof ModelFormatBadge>[0]['format'];
  imageVersion?: number;
  isMenuOpen: boolean;
  isMissing: boolean;
  isSelected: boolean;
  modelKey: string;
  name: string;
  onActivate: (modelKey: string) => void;
  onContextMenu: (modelKey: string, anchor: ListContextMenuAnchor) => void;
  onToggleSelected: (modelKey: string) => void;
}

const ModelRow = memo(function ModelRow({
  base,
  coverImage,
  fileSize,
  format,
  imageVersion,
  isMissing,
  isSelected,
  modelKey,
  name,
  onActivate,
  onContextMenu,
  onToggleSelected,
  ...rowProps
}: ModelRowProps) {
  const handlePress = useCallback(() => onActivate(modelKey), [modelKey, onActivate]);
  const handleCheckedChange = useCallback(() => onToggleSelected(modelKey), [modelKey, onToggleSelected]);
  const handleContextMenu = useCallback(
    (anchor: ListContextMenuAnchor) => onContextMenu(modelKey, anchor),
    [modelKey, onContextMenu]
  );

  return (
    <ListItem
      {...rowProps}
      description={
        <HStack gap="1" minW="0" wrap="wrap">
          <ModelBaseBadge base={base} />
          <ModelFormatBadge format={format} />
          {isMissing ? <MissingFileBadge /> : null}
        </HStack>
      }
      isChecked={isSelected}
      leading={
        // Keyed by version so a stale load error clears when the image changes.
        <ModelRowThumbnail
          key={`${modelKey}:${imageVersion ?? 0}`}
          coverImage={coverImage}
          imageVersion={imageVersion}
          modelKey={modelKey}
        />
      }
      title={name}
      trailing={formatBytes(fileSize)}
      onCheckedChange={handleCheckedChange}
      onContextMenu={handleContextMenu}
      onPress={handlePress}
    />
  );
});

const ModelRowThumbnail = ({
  coverImage,
  imageVersion,
  modelKey,
}: {
  coverImage?: string | null;
  imageVersion?: number;
  modelKey: string;
}) => {
  // The cover_image marker can be stale; fall back to the icon on load error.
  const [hasImageError, setHasImageError] = useState(false);

  if (!coverImage || hasImageError) {
    return (
      <Flex
        align="center"
        bg="bg.emphasized"
        borderColor="border.subtle"
        borderWidth="1px"
        boxSize="9"
        color="fg.subtle"
        flexShrink={0}
        justify="center"
        rounded="md"
      >
        <Icon as={BoxIcon} boxSize="4" />
      </Flex>
    );
  }

  return (
    <Image
      alt=""
      boxSize="9"
      fit="cover"
      flexShrink={0}
      loading="lazy"
      rounded="md"
      src={getModelImageUrl(modelKey, imageVersion ? String(imageVersion) : undefined)}
      onError={() => setHasImageError(true)}
    />
  );
};
