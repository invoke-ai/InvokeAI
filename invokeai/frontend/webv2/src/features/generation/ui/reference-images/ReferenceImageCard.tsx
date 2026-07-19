import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
} from '@features/generation/core/types';
import type { CSSProperties } from 'react';

import { Box, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { getEffectiveReferenceImage, getReferenceImageUrls } from '@features/generation/core/referenceImage';
import { GenerationModelSelect as ModelSelect } from '@features/generation/ui/GenerationUiContext';
import { IconButton, ToggleDot, Tooltip } from '@platform/ui';
import { ChevronDownIcon, CropIcon, ImageIcon, RulerIcon, Trash2Icon } from 'lucide-react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatWeight, getModeLabelKey } from './referenceImageConfig';
import { ClipVisionSelect, FieldHeader, FluxReduxControls, IPAdapterControls } from './ReferenceImageControls';
import { ReferenceImageCropDialog } from './ReferenceImageCropDialog';

/** Reference image config variants that carry a selectable model. */
type ModelBackedConfig = Extract<
  GenerateReferenceImageConfig,
  { type: 'ip_adapter' | 'flux_redux' | 'flux_kontext_reference_image' }
>;

const THUMBNAIL_ACTIONS_CSS = {
  '&:focus-within .reference-image-actions, &:hover .reference-image-actions': {
    opacity: 1,
  },
};

const FLUX_MODEL_TYPES = ['ip_adapter', 'flux_redux'];
const DEFAULT_MODEL_TYPES = ['ip_adapter'];
const COVER_IMG_STYLE: CSSProperties = {
  height: '100%',
  objectFit: 'cover',
  width: '100%',
};
const OVERLAY_GRADIENT_STYLE: CSSProperties = {
  background: 'linear-gradient(to top, rgba(0, 0, 0, 0.72), transparent)',
};

interface ReferenceImageCardProps {
  index: number;
  referenceImage: GenerateReferenceImage;
  selectedModel: GenerateModelConfig | undefined;
  onPatch: (id: string, patch: Partial<GenerateReferenceImage>) => void;
  onRemove: (id: string) => void;
  onUseSize: (image: GenerateReferenceImageAsset) => void;
}

const ReferenceImageCardBase = ({
  index,
  onPatch,
  onRemove,
  onUseSize,
  referenceImage,
  selectedModel,
}: ReferenceImageCardProps) => {
  const { t } = useTranslation();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const config = referenceImage.config;
  const isEnabled = referenceImage.isEnabled;
  const selectedBase = selectedModel?.base;
  const title = `${t('widgets.generate.referenceImage')} #${index + 1}`;
  const toggleLabel = isEnabled
    ? t('widgets.generate.disableReferenceImage')
    : t('widgets.generate.enableReferenceImage');

  const changeConfig = useCallback(
    (nextConfig: GenerateReferenceImageConfig) => onPatch(referenceImage.id, { config: nextConfig }),
    [onPatch, referenceImage.id]
  );

  const handleToggle = useCallback(
    (nextEnabled: boolean) => onPatch(referenceImage.id, { isEnabled: nextEnabled }),
    [onPatch, referenceImage.id]
  );

  const handleRemove = useCallback(() => onRemove(referenceImage.id), [onRemove, referenceImage.id]);

  const handleCrop = useCallback(
    (image: GenerateReferenceImageAsset) => changeConfig({ ...config, image }),
    [changeConfig, config]
  );

  const expand = useCallback(() => setIsCollapsed(false), []);

  const toggleCollapsed = useCallback(() => setIsCollapsed((previous) => !previous), []);

  return (
    <Stack
      bg="bg.subtle"
      borderColor={isEnabled ? 'border.emphasized' : 'border.subtle'}
      borderWidth="1px"
      gap="0"
      opacity={isEnabled ? 1 : 0.55}
      rounded="md"
      transition="opacity var(--wb-motion-duration-slow)"
    >
      <HStack gap="2" minW="0" px="2" py="1.5">
        <ToggleDot checked={isEnabled} label={toggleLabel} onCheckedChange={handleToggle} />

        {isCollapsed ? (
          <HStack as="button" cursor="pointer" flex="1" gap="2" minW="0" textAlign="left" onClick={expand}>
            <MiniThumbnail image={config.image} />
            <Text color="fg.muted" fontSize="xs" minW="0" truncate>
              <Text as="span" color={isEnabled ? 'fg' : 'fg.muted'} fontWeight="medium">
                {title}
              </Text>
              {config.type === 'ip_adapter' ? (
                <>
                  {' · '}
                  {t(getModeLabelKey(config.method))}{' '}
                  <Text as="span" color="fg.subtle" fontFamily="mono">
                    {formatWeight(config.weight)}
                  </Text>
                </>
              ) : null}
            </Text>
          </HStack>
        ) : (
          <Text color={isEnabled ? 'fg' : 'fg.muted'} flex="1" fontSize="xs" fontWeight="medium" minW="0" truncate>
            {title}
          </Text>
        )}

        <HStack gap="0.5">
          <IconButton
            aria-label={
              isCollapsed ? t('widgets.generate.expandReferenceImage') : t('widgets.generate.collapseReferenceImage')
            }
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={toggleCollapsed}
          >
            <Box
              asChild
              transform={isCollapsed ? 'rotate(-90deg)' : undefined}
              transition="transform var(--wb-motion-duration-slow)"
            >
              <ChevronDownIcon size="14" />
            </Box>
          </IconButton>
          <Tooltip content={t('widgets.generate.removeReferenceImage')}>
            <IconButton
              aria-label={t('widgets.generate.removeReferenceImage')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              colorPalette="red"
              onClick={handleRemove}
            >
              <Icon as={Trash2Icon} colorPalette="red" />
            </IconButton>
          </Tooltip>
        </HStack>
      </HStack>

      {isCollapsed ? null : (
        <HStack align="flex-start" gap="3" pb="2" px="2">
          <ReferenceImageThumbnail
            disabled={!isEnabled}
            image={config.image}
            onCrop={handleCrop}
            onUseSize={onUseSize}
          />
          <Stack flex="1" gap="2" minW="0">
            {config.type === 'ip_adapter' ? (
              <IPAdapterControls
                key={config.model?.key ?? 'no-model'}
                config={config}
                disabled={!isEnabled}
                onChange={changeConfig}
              >
                <ReferenceModelSelector
                  config={config}
                  disabled={!isEnabled}
                  selectedBase={selectedBase}
                  onConfigChange={changeConfig}
                />
              </IPAdapterControls>
            ) : config.type === 'flux_redux' ? (
              <>
                <ReferenceModelSelector
                  config={config}
                  disabled={!isEnabled}
                  selectedBase={selectedBase}
                  onConfigChange={changeConfig}
                />
                <FluxReduxControls config={config} disabled={!isEnabled} onChange={changeConfig} />
              </>
            ) : config.type === 'flux_kontext_reference_image' ? (
              <ReferenceModelSelector
                config={config}
                disabled={!isEnabled}
                selectedBase={selectedBase}
                onConfigChange={changeConfig}
              />
            ) : null}
          </Stack>
        </HStack>
      )}
    </Stack>
  );
};

export const ReferenceImageCard = memo(ReferenceImageCardBase);

const ReferenceModelSelector = ({
  config,
  disabled,
  onConfigChange,
  selectedBase,
}: {
  config: ModelBackedConfig;
  disabled: boolean;
  selectedBase: string | undefined;
  onConfigChange: (config: GenerateReferenceImageConfig) => void;
}) => {
  const { t } = useTranslation();
  const modelTypes = selectedBase === 'flux' ? FLUX_MODEL_TYPES : DEFAULT_MODEL_TYPES;

  const filterModel = useCallback((model: ModelConfig) => model.base === selectedBase, [selectedBase]);

  const selectReferenceModel = useCallback(
    (model: ModelConfig | null) => {
      if (!model || model.base !== selectedBase) {
        return;
      }

      if (model.type === 'flux_redux') {
        onConfigChange({
          image: config.image,
          imageInfluence: 'highest',
          model,
          type: 'flux_redux',
        });
        return;
      }

      if (model.type === 'ip_adapter') {
        onConfigChange({
          beginEndStepPct: [0, 1],
          clipVisionModel: selectedBase === 'flux' ? 'ViT-L' : 'ViT-H',
          image: config.image,
          method: 'full',
          model,
          type: 'ip_adapter',
          weight: 1,
        });
      }
    },
    [config.image, onConfigChange, selectedBase]
  );

  return (
    <Stack gap="1">
      <FieldHeader label={t('widgets.generate.referenceImageModel')} />
      <HStack gap="2">
        <ModelSelect
          disabled={disabled}
          filter={filterModel}
          modelTypes={modelTypes}
          placeholder={t('widgets.generate.selectModel')}
          size="xs"
          value={config.model?.key ?? null}
          onChange={selectReferenceModel}
        />
        {config.type === 'ip_adapter' ? (
          <ClipVisionSelect
            key={config.model?.key ?? 'no-model'}
            config={config}
            disabled={disabled}
            onChange={onConfigChange}
          />
        ) : null}
      </HStack>
    </Stack>
  );
};

const MiniThumbnail = ({ image }: { image: GenerateReferenceImageAsset | null }) => {
  const effectiveImage = image ? getEffectiveReferenceImage(image) : null;
  const urls = image ? getReferenceImageUrls(image) : null;

  return (
    <Box bg="bg.muted" borderWidth="1px" flexShrink="0" h="6" overflow="hidden" rounded="sm" w="6">
      {effectiveImage && urls ? (
        <img alt={effectiveImage.image_name} draggable={false} src={urls.thumbnailUrl} style={COVER_IMG_STYLE} />
      ) : (
        <Box alignItems="center" color="fg.muted" display="flex" h="full" justifyContent="center" w="full">
          <ImageIcon size="12" />
        </Box>
      )}
    </Box>
  );
};

const ReferenceImageThumbnail = ({
  disabled,
  image,
  onCrop,
  onUseSize,
}: {
  disabled: boolean;
  image: GenerateReferenceImageAsset | null;
  onCrop: (image: GenerateReferenceImageAsset) => void;
  onUseSize: (image: GenerateReferenceImageAsset) => void;
}) => {
  const { t } = useTranslation();
  const [isCropOpen, setIsCropOpen] = useState(false);
  const effectiveImage = image ? getEffectiveReferenceImage(image) : null;
  const urls = image ? getReferenceImageUrls(image) : null;

  const openCrop = useCallback(() => setIsCropOpen(true), []);
  const closeCrop = useCallback(() => setIsCropOpen(false), []);

  const handleUseSize = useCallback(() => {
    if (image) {
      onUseSize(image);
    }
  }, [image, onUseSize]);

  return (
    <>
      <Box
        bg="bg.muted"
        borderWidth="1px"
        css={THUMBNAIL_ACTIONS_CSS}
        flexShrink="0"
        h="20"
        overflow="hidden"
        position="relative"
        rounded="md"
        w="20"
      >
        {effectiveImage && urls ? (
          <img alt={effectiveImage.image_name} draggable={false} src={urls.thumbnailUrl} style={COVER_IMG_STYLE} />
        ) : (
          <Box alignItems="center" color="fg.muted" display="flex" h="full" justifyContent="center" w="full">
            <ImageIcon size="24" />
          </Box>
        )}
        {image ? (
          <HStack
            bottom="0"
            className="reference-image-actions"
            gap="1"
            justify="center"
            left="0"
            opacity="0"
            p="1"
            position="absolute"
            right="0"
            style={OVERLAY_GRADIENT_STYLE}
            transition="opacity var(--wb-motion-duration-fast)"
          >
            <Tooltip content={t('common.crop')}>
              <IconButton
                aria-label={t('common.crop')}
                disabled={disabled}
                size="2xs"
                variant="ghost"
                onClick={openCrop}
              >
                <CropIcon />
              </IconButton>
            </Tooltip>
            <Tooltip content={t('widgets.generate.useSize')}>
              <IconButton
                aria-label={t('widgets.generate.useSize')}
                disabled={disabled}
                size="2xs"
                variant="ghost"
                onClick={handleUseSize}
              >
                <RulerIcon />
              </IconButton>
            </Tooltip>
          </HStack>
        ) : null}
      </Box>
      {image && isCropOpen ? (
        <ReferenceImageCropDialog image={image} isOpen={isCropOpen} onApply={onCrop} onClose={closeCrop} />
      ) : null}
    </>
  );
};
