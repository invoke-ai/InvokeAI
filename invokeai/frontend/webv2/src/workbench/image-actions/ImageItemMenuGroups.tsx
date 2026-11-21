import type { GalleryImage, GalleryVideoItem } from '@features/gallery';
import type { ModelsSnapshot } from '@features/models';
import type { SpandrelModelConfig } from '@features/upscale';
import type { VideoConditioningRole } from '@features/video';
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';

import { Menu, Text } from '@chakra-ui/react';
import { getImageExportSummary } from '@features/gallery/contracts';
import { galleryVideoMetadataOptions } from '@features/gallery/queries';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { getModelOnlyUpscaleFactors, getSpandrelNativeScale } from '@features/upscale';
import { formatCount } from '@platform/i18n/languages';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { MenuActionItem, MenuItem } from '@platform/ui/Menu';
import { useQuery } from '@tanstack/react-query';
import { ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { useShortcutBinding } from '@workbench/hotkeys/useShortcutBinding';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { LAYER_KIND_ICONS } from '@workbench/widgets/layers/layerTypeIcon';
import { useActiveProjectId, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import {
  AsteriskIcon,
  AudioLinesIcon,
  ClapperboardIcon,
  FileImageIcon,
  FilmIcon,
  ImageIcon,
  ImagePlayIcon,
  ImageUpIcon,
  LayersIcon,
  PencilIcon,
  QuoteIcon,
  RulerIcon,
  ScalingIcon,
  ScanIcon,
  ScissorsIcon,
  ShuffleIcon,
  SlidersHorizontalIcon,
  SproutIcon,
  TypeIcon,
  ZapIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ImageExportTarget } from './imageExport';
import type { ImageRecallCapabilities, ImageRecallKind } from './imageRecall';
import type { ImageActions } from './useImageActions';

import { ContextSubMenu } from './ContextSubMenu';
import { EXPORT_IMAGE_COMMAND_ID, useImageExportPreference } from './imageExport';
import {
  EMPTY_VIDEO_RECALL_CAPABILITIES,
  getVideoRecallCapabilities,
  type VideoRecallCapabilities,
  type VideoRecallKind,
} from './videoRecall';

/* Shared by the Gallery context menu and Preview's Edit and Use menus, so their items and order cannot drift apart. */

/** `guided` adds hints, keycaps and full settings for a menu of these groups alone; `compact` folds settings away. */
export type ImageMenuPresentation = 'compact' | 'guided';

interface GalleryCanvasImportDestinationOption {
  destination: GalleryCanvasImportDestination;
  icon: LucideIcon;
  label: string;
}

/** A resized control layer is still a control layer, so it takes a resize glyph to tell the two apart. */
export const GALLERY_CANVAS_IMPORT_DESTINATIONS: readonly GalleryCanvasImportDestinationOption[] = [
  { destination: 'raster', icon: LAYER_KIND_ICONS.raster, label: 'widgets.canvas.import.raster' },
  { destination: 'control', icon: LAYER_KIND_ICONS.control, label: 'widgets.canvas.import.control' },
  { destination: 'inpaint-mask', icon: LAYER_KIND_ICONS.inpaint_mask, label: 'widgets.canvas.import.inpaintMask' },
  {
    destination: 'regional-guidance',
    icon: LAYER_KIND_ICONS.regional_guidance,
    label: 'widgets.canvas.import.regionalGuidance',
  },
  {
    destination: 'regional-reference',
    icon: LAYER_KIND_ICONS.regional_reference_image,
    label: 'widgets.canvas.import.regionalReference',
  },
  { destination: 'control-resized', icon: ScalingIcon, label: 'widgets.canvas.import.resizedControl' },
];

const GalleryCanvasImportMenuItems = ({
  onSelect,
}: {
  onSelect: (destination: GalleryCanvasImportDestination) => void;
}) => (
  <>
    {GALLERY_CANVAS_IMPORT_DESTINATIONS.map((option) => (
      <GalleryCanvasImportMenuItem key={option.destination} option={option} onSelect={onSelect} />
    ))}
  </>
);

const GalleryCanvasImportMenuItem = ({
  option,
  onSelect,
}: {
  option: GalleryCanvasImportDestinationOption;
  onSelect: (destination: GalleryCanvasImportDestination) => void;
}) => {
  const { t } = useTranslation();
  const handleSelect = useCallback(() => onSelect(option.destination), [onSelect, option.destination]);

  return (
    <MenuActionItem
      icon={option.icon}
      label={t(option.label)}
      value={`send-to-canvas-${option.destination}`}
      onSelect={handleSelect}
    />
  );
};

/** A selection of several images keeps only the canvas items, which take them all. */
export const ImageContinueGroup = ({
  actions,
  images,
  presentation,
}: {
  actions: ImageActions;
  images: readonly GalleryImage[];
  presentation: ImageMenuPresentation;
}) => {
  const { t } = useTranslation();
  const guided = presentation === 'guided';
  const image = images.length === 1 ? images[0]! : null;
  const handleSend = useCallback(
    (destination: GalleryCanvasImportDestination) => void actions.sendToCanvas(images, destination),
    [actions, images]
  );
  const handleNewProject = useCallback(() => void actions.createCanvasFromImages([...images]), [actions, images]);

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel>{t('widgets.gallery.itemMenu.continueWorking')}</Menu.ItemGroupLabel>
      {image ? (
        <>
          <SingleImageHandOffItems actions={actions} guided={guided} image={image} />
          <Menu.Separator borderColor="border.subtle" />
        </>
      ) : null}
      <ContextSubMenu
        hint={guided ? t('widgets.gallery.itemMenu.addToCanvasAsHint') : undefined}
        icon={LayersIcon}
        label={t('widgets.gallery.itemMenu.addToCanvasAs')}
      >
        <GalleryCanvasImportMenuItems onSelect={handleSend} />
      </ContextSubMenu>
      <MenuActionItem
        hint={guided ? t('widgets.gallery.itemMenu.newProjectFromImageHint', { count: images.length }) : undefined}
        icon={FileImageIcon}
        label={t('widgets.canvas.import.newProjectFromImage', { count: images.length })}
        value="new-project-from-image"
        onSelect={handleNewProject}
      />
    </Menu.ItemGroup>
  );
};

const SingleImageHandOffItems = ({
  actions,
  guided,
  image,
}: {
  actions: ImageActions;
  guided: boolean;
  image: GalleryImage;
}) => {
  const { t } = useTranslation();
  const { generation, widgets } = useWorkbenchCommands();
  const openWidget = useOpenWorkbenchWidget();
  const handleEditOnCanvas = useCallback(() => void actions.sendToCanvas([image], 'raster'), [actions, image]);
  const handleSendToUpscale = useCallback(() => {
    openWidget('upscale', { preferredRegions: ['left'] });
    widgets.patchValues('upscale', {
      inputImage: { height: image.height, image_name: image.imageName, width: image.width },
    });
    generation.setSource('upscale');
  }, [generation, image.height, image.imageName, image.width, openWidget, widgets]);
  const handleSendToVideo = useCallback(() => {
    openWidget('video', { preferredRegions: ['left'] });
    // A frame, an initial video and Ref2VA references claim one conditioning slot: a frame clears the rivals.
    widgets.patchValues('video', {
      firstFrameImage: { height: image.height, image_name: image.imageName, width: image.width },
      references: [],
      sourceVideo: null,
    });
    generation.setSource('video');
  }, [generation, image.height, image.imageName, image.width, openWidget, widgets]);

  return (
    <>
      <MenuActionItem
        hint={guided ? t('widgets.gallery.itemMenu.editOnCanvasHint') : undefined}
        icon={PencilIcon}
        label={t('widgets.preview.editOnCanvas')}
        value="edit-on-canvas"
        onSelect={handleEditOnCanvas}
      />
      <UpscaleSubMenu guided={guided} image={image} onSendToUpscale={handleSendToUpscale} />
      <MenuActionItem
        hint={guided ? t('widgets.gallery.itemMenu.sendToVideoHint') : undefined}
        icon={ClapperboardIcon}
        label={t('widgets.gallery.itemMenu.sendToVideo')}
        value="send-to-video"
        onSelect={handleSendToVideo}
      />
    </>
  );
};

interface ReuseItem {
  capability: keyof ImageRecallCapabilities;
  /** None where the viewer has no key. */
  commandId?: string;
  hint?: string;
  icon: LucideIcon;
  kind: ImageRecallKind;
  label: string;
  value: string;
}

/** Remix leads; the full recall, which keeps the seed, follows. */
const REUSE_ITEMS: readonly ReuseItem[] = [
  {
    capability: 'remix',
    commandId: 'viewer.remix',
    hint: 'widgets.gallery.itemMenu.remixHint',
    icon: ShuffleIcon,
    kind: 'remix',
    label: 'widgets.gallery.itemActions.recall.verbs.remix',
    value: 'remix',
  },
  {
    capability: 'all',
    commandId: 'viewer.recallAll',
    hint: 'widgets.gallery.itemMenu.allSettingsHint',
    icon: AsteriskIcon,
    kind: 'all',
    label: 'widgets.gallery.itemMenu.allSettings',
    value: 'recall-all',
  },
  {
    capability: 'prompts',
    commandId: 'viewer.recallPrompts',
    icon: QuoteIcon,
    kind: 'prompts',
    label: 'widgets.gallery.itemActions.recall.verbs.prompts',
    value: 'use-prompt',
  },
  {
    capability: 'seed',
    commandId: 'viewer.recallSeed',
    icon: SproutIcon,
    kind: 'seed',
    label: 'widgets.gallery.itemActions.recall.verbs.seed',
    value: 'use-seed',
  },
  {
    capability: 'dimensions',
    commandId: 'viewer.useSize',
    icon: RulerIcon,
    kind: 'dimensions',
    label: 'widgets.gallery.itemActions.recall.verbs.dimensions',
    value: 'use-size',
  },
  {
    capability: 'clipSkip',
    icon: ScissorsIcon,
    kind: 'clipSkip',
    label: 'widgets.gallery.itemActions.recall.verbs.clipSkip',
    value: 'use-clip-skip',
  },
];

export const ImageUseGroups = ({
  actions,
  capabilities,
  image,
  isLoadingCapabilities,
  presentation,
}: {
  actions: ImageActions;
  capabilities: ImageRecallCapabilities;
  image: GalleryImage;
  isLoadingCapabilities: boolean;
  presentation: ImageMenuPresentation;
}) => {
  const { t } = useTranslation();
  const guided = presentation === 'guided';
  // eslint-disable-next-line react/hooks -- `useAsReferenceImage` is a plain action method; the compiler reads its name as a hook.
  const handleUseAsReferenceImage = useCallback(() => actions.useAsReferenceImage(image), [actions, image]);
  const handleUseAsControl = useCallback(() => void actions.sendImageAsControl(image), [actions, image]);
  const handleSavePromptAsTemplate = useCallback(() => void actions.savePromptAsTemplate(image), [actions, image]);
  const reuseItems = REUSE_ITEMS.map((item) => (
    <ReuseMenuItem
      key={item.kind}
      actions={actions}
      disabled={isLoadingCapabilities || !capabilities[item.capability]}
      guided={guided}
      image={image}
      item={item}
    />
  ));

  return (
    <>
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('widgets.gallery.itemMenu.useImageAs')}</Menu.ItemGroupLabel>
        <MenuActionItem
          disabled={!actions.canUseAsReferenceImage}
          hint={guided ? t('widgets.gallery.itemMenu.useAsReferenceImageHint') : undefined}
          icon={ImageIcon}
          label={t('widgets.gallery.itemMenu.useAsReferenceImage')}
          value="use-as-reference-image"
          onSelect={handleUseAsReferenceImage}
        />
        <MenuActionItem
          hint={guided ? t('widgets.gallery.itemMenu.useAsControlHint') : undefined}
          icon={SlidersHorizontalIcon}
          label={t('widgets.gallery.itemActions.useAsControl')}
          value="use-as-control"
          onSelect={handleUseAsControl}
        />
        <MenuActionItem
          disabled={isLoadingCapabilities || !capabilities.prompts}
          hint={guided ? t('widgets.gallery.itemMenu.useAsPromptTemplateHint') : undefined}
          icon={TypeIcon}
          label={t('widgets.gallery.itemMenu.useAsPromptTemplate')}
          value="use-as-prompt-template"
          onSelect={handleSavePromptAsTemplate}
        />
      </Menu.ItemGroup>
      {/* Outside the group, so assistive technology does not announce it under that heading. */}
      {guided ? null : (
        <ContextSubMenu icon={AsteriskIcon} label={t('widgets.gallery.itemMenu.reuseFromImage')}>
          {reuseItems}
        </ContextSubMenu>
      )}
      {guided ? (
        <>
          <Menu.Separator borderColor="border.subtle" />
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel>{t('widgets.gallery.itemMenu.reuseFromImage')}</Menu.ItemGroupLabel>
            {reuseItems}
          </Menu.ItemGroup>
        </>
      ) : null}
    </>
  );
};

const ReuseMenuItem = ({
  actions,
  disabled,
  guided,
  image,
  item,
}: {
  actions: ImageActions;
  disabled: boolean;
  guided: boolean;
  image: GalleryImage;
  item: ReuseItem;
}) => {
  const { t } = useTranslation();
  const handleSelect = useCallback(() => void actions.recallImageData(image, item.kind), [actions, image, item.kind]);
  const shortcut = useMemo(
    () => (guided && item.commandId ? <CommandKeycaps commandId={item.commandId} /> : undefined),
    [guided, item.commandId]
  );

  return (
    <MenuActionItem
      disabled={disabled}
      hint={guided && item.hint ? t(item.hint) : undefined}
      icon={item.icon}
      label={t(item.label)}
      shortcut={shortcut}
      value={item.value}
      onSelect={handleSelect}
    />
  );
};

interface VideoReuseItem {
  capability: keyof VideoRecallCapabilities;
  commandId: string;
  icon: LucideIcon;
  kind: VideoRecallKind;
  label: string;
  value: string;
}

const VIDEO_REUSE_ITEMS: readonly VideoReuseItem[] = [
  {
    capability: 'remix',
    commandId: 'viewer.remix',
    icon: ShuffleIcon,
    kind: 'remix',
    label: 'widgets.gallery.itemMenu.remixVideo',
    value: 'video-remix',
  },
  {
    capability: 'all',
    commandId: 'viewer.recallAll',
    icon: AsteriskIcon,
    kind: 'all',
    label: 'widgets.gallery.itemMenu.allSettings',
    value: 'video-recall-all',
  },
  {
    capability: 'prompts',
    commandId: 'viewer.recallPrompts',
    icon: QuoteIcon,
    kind: 'prompts',
    label: 'widgets.gallery.itemActions.recall.verbs.prompts',
    value: 'video-use-prompt',
  },
  {
    capability: 'seed',
    commandId: 'viewer.recallSeed',
    icon: SproutIcon,
    kind: 'seed',
    label: 'widgets.gallery.itemActions.recall.verbs.seed',
    value: 'video-use-seed',
  },
  {
    capability: 'size',
    commandId: 'viewer.useSize',
    icon: RulerIcon,
    kind: 'size',
    label: 'widgets.gallery.itemActions.recall.verbs.dimensions',
    value: 'video-use-size',
  },
];

/** Reads the shared metadata entry on mount, so hosts mount it with their open menu; failures retry next open. */
export const VideoUseGroups = ({
  actions,
  item,
  presentation,
}: {
  actions: ImageActions;
  item: GalleryVideoItem;
  presentation: ImageMenuPresentation;
}) => {
  const { t } = useTranslation();
  const metadataQuery = useQuery(galleryVideoMetadataOptions(captureAccountScope(), item.name));
  const capabilities = metadataQuery.isSuccess
    ? getVideoRecallCapabilities(metadataQuery.data)
    : EMPTY_VIDEO_RECALL_CAPABILITIES;
  const handleExtendInVideo = useCallback(() => actions.sendToInitialVideo(item), [actions, item]);
  // eslint-disable-next-line react/hooks -- `useAsReferenceVideo` is a plain action method; the compiler reads its name as a hook.
  const handleUseAsReferenceVideo = useCallback(() => actions.useAsReferenceVideo(item), [actions, item]);
  const reuseItems = VIDEO_REUSE_ITEMS.map((reuseItem) => (
    <VideoReuseMenuItem
      key={reuseItem.kind}
      actions={actions}
      disabled={metadataQuery.isPending || !capabilities[reuseItem.capability]}
      guided={presentation === 'guided'}
      item={item}
      reuseItem={reuseItem}
    />
  ));

  return (
    <>
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('widgets.gallery.itemMenu.useVideoAs')}</Menu.ItemGroupLabel>
        <MenuActionItem
          icon={ClapperboardIcon}
          label={t('widgets.gallery.itemMenu.extendInVideo')}
          value="extend-in-video"
          onSelect={handleExtendInVideo}
        />
        <MenuActionItem
          disabled={!actions.canUseAsReferenceVideo}
          icon={FilmIcon}
          label={t('widgets.gallery.itemMenu.useAsReferenceVideo')}
          value="use-as-reference-video"
          onSelect={handleUseAsReferenceVideo}
        />
        <ContextSubMenu icon={AudioLinesIcon} label={t('widgets.gallery.itemMenu.useAsConditioningClip')}>
          <ConditioningClipMenuItem actions={actions} conditioningRole="audio" item={item} />
          <ConditioningClipMenuItem actions={actions} conditioningRole="video" item={item} />
        </ContextSubMenu>
      </Menu.ItemGroup>
      {/* Outside the group, as for images. */}
      {presentation === 'guided' ? (
        <>
          <Menu.Separator borderColor="border.subtle" />
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel>{t('widgets.gallery.itemMenu.reuseFromVideo')}</Menu.ItemGroupLabel>
            {reuseItems}
          </Menu.ItemGroup>
        </>
      ) : (
        <ContextSubMenu icon={AsteriskIcon} label={t('widgets.gallery.itemMenu.reuseFromVideo')}>
          {reuseItems}
        </ContextSubMenu>
      )}
    </>
  );
};

const ConditioningClipMenuItem = ({
  actions,
  conditioningRole: role,
  item,
}: {
  actions: ImageActions;
  conditioningRole: VideoConditioningRole;
  item: GalleryVideoItem;
}) => {
  const { t } = useTranslation();
  // eslint-disable-next-line react/hooks -- `useAsConditioningClip` is a plain action method; the compiler reads its name as a hook.
  const handleSelect = useCallback(() => actions.useAsConditioningClip(item, role), [actions, item, role]);

  return (
    <MenuActionItem
      // A wrapped audio upload's picture is a placeholder.
      disabled={!actions.canUseAsConditioningClip[role] || (role === 'video' && item.mediaOrigin === 'audio_upload')}
      icon={role === 'audio' ? AudioLinesIcon : ImagePlayIcon}
      label={t(role === 'audio' ? 'widgets.video.conditioningRoleAudio' : 'widgets.video.conditioningRoleVideo')}
      value={`use-as-conditioning-${role}`}
      onSelect={handleSelect}
    />
  );
};

const VideoReuseMenuItem = ({
  actions,
  disabled,
  guided,
  item,
  reuseItem,
}: {
  actions: ImageActions;
  disabled: boolean;
  guided: boolean;
  item: GalleryVideoItem;
  reuseItem: VideoReuseItem;
}) => {
  const { t } = useTranslation();
  const handleSelect = useCallback(
    () => void actions.recallVideoData(item, reuseItem.kind),
    [actions, item, reuseItem.kind]
  );
  const shortcut = useMemo(
    () => (guided ? <CommandKeycaps commandId={reuseItem.commandId} /> : undefined),
    [guided, reuseItem.commandId]
  );

  return (
    <MenuActionItem
      disabled={disabled}
      icon={reuseItem.icon}
      label={t(reuseItem.label)}
      shortcut={shortcut}
      value={reuseItem.value}
      onSelect={handleSelect}
    />
  );
};

/** Export with the dialog, or at once with the settings it last saved; a selection without images offers neither. */
export const ImageExportGroup = ({
  actions,
  presentation,
  returnFocus,
  target,
}: {
  actions: ImageActions;
  presentation: ImageMenuPresentation;
  returnFocus?: () => HTMLElement | null;
  target: ImageExportTarget;
}) => {
  const { t } = useTranslation();
  const settings = useImageExportPreference();
  const guided = presentation === 'guided';
  const isSelection = target.kind === 'selection';
  const unavailableReason =
    isSelection && target.imageNames.length === 0 ? t('widgets.gallery.export.noImagesInSelection') : undefined;
  const summary = getImageExportSummary(settings, (pixels) =>
    t('widgets.gallery.export.longEdgeSummary', { size: formatCount(pixels) })
  );
  const handleExport = useCallback(
    () => actions.openImageExport(target, { returnFocus }),
    [actions, returnFocus, target]
  );
  const handleQuickExport = useCallback(() => void actions.quickExportImages(target), [actions, target]);
  const shortcut = useMemo(
    () => (guided && !isSelection ? <CommandKeycaps commandId={EXPORT_IMAGE_COMMAND_ID} /> : undefined),
    [guided, isSelection]
  );
  const items = (
    <>
      <MenuActionItem
        icon={ImageUpIcon}
        label={t(isSelection ? 'widgets.gallery.export.exportSelectionMenu' : 'widgets.gallery.export.exportMenu')}
        shortcut={shortcut}
        unavailableReason={unavailableReason}
        value="export-image"
        onSelect={handleExport}
      />
      <MenuActionItem
        icon={ZapIcon}
        label={t('widgets.gallery.export.quickExport', { summary })}
        unavailableReason={unavailableReason}
        value="quick-export"
        onSelect={handleQuickExport}
      />
    </>
  );

  return guided ? (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel>{t('widgets.gallery.export.menuGroup')}</Menu.ItemGroupLabel>
      {items}
    </Menu.ItemGroup>
  ) : (
    items
  );
};

const CommandKeycaps = ({ commandId }: { commandId: string }) => {
  const binding = useShortcutBinding(commandId);

  return binding ? <ShortcutKeycaps parts={binding.parts} /> : null;
};

/** A shallow-equal list keeps the menu from rerendering on unrelated model edits. */
const selectInstalledUpscalers = (snapshot: ModelsSnapshot): SpandrelModelConfig[] =>
  snapshot.models
    .filter((model) => model.type === 'spandrel_image_to_image')
    .map((model) => model as SpandrelModelConfig)
    .sort((a, b) => a.name.localeCompare(b.name));

const selectModelsStatus = (snapshot: ModelsSnapshot): ModelsSnapshot['status'] => snapshot.status;

/** The items below Send to Upscale upscale at once into the gallery, leaving the widget as it is. */
const UpscaleSubMenu = ({
  guided,
  image,
  onSendToUpscale,
}: {
  guided: boolean;
  image: GalleryImage;
  onSendToUpscale: () => void;
}) => {
  const { t } = useTranslation();
  const upscalers = useModelsSelector(selectInstalledUpscalers);
  const modelsStatus = useModelsSelector(selectModelsStatus);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  return (
    <ContextSubMenu
      hint={guided ? t('widgets.gallery.itemMenu.upscaleHint') : undefined}
      icon={ScanIcon}
      label={t('widgets.gallery.itemActions.upscale.label')}
    >
      <MenuActionItem
        icon={ScanIcon}
        label={t('widgets.gallery.itemActions.upscale.sendTo')}
        value="send-to-upscale"
        onSelect={onSendToUpscale}
      />
      {upscalers.length === 0 ? (
        // "None installed" only once the library has loaded.
        <MenuItem
          unavailableReason={
            modelsStatus === 'loaded'
              ? t('widgets.gallery.itemActions.upscale.noUpscalers')
              : modelsStatus === 'error'
                ? t('models.couldNotLoad')
                : t('models.loadingModels')
          }
          value="upscale-model-only"
        >
          <Text fontSize="md">{t('widgets.gallery.itemActions.upscale.modelOnlyGroup')}</Text>
        </MenuItem>
      ) : (
        <>
          <Menu.Separator borderColor="border.subtle" />
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel>{t('widgets.gallery.itemActions.upscale.modelOnlyGroup')}</Menu.ItemGroupLabel>
            {upscalers.map((upscaler) => (
              <ModelOnlyUpscalerSubMenu key={upscaler.key} image={image} upscaler={upscaler} />
            ))}
          </Menu.ItemGroup>
        </>
      )}
    </ContextSubMenu>
  );
};
/** One whose native scale was never recorded can still run once at it; other factors say how to record it. */
const ModelOnlyUpscalerSubMenu = ({ image, upscaler }: { image: GalleryImage; upscaler: SpandrelModelConfig }) => {
  const { t } = useTranslation();
  const factors = getModelOnlyUpscaleFactors(upscaler);
  const nativeScale = getSpandrelNativeScale(upscaler);

  return (
    <ContextSubMenu
      detail={
        nativeScale === null ? undefined : t('widgets.gallery.itemActions.upscale.scaleSuffix', { scale: nativeScale })
      }
      label={upscaler.name}
    >
      {factors ? (
        factors.map((factor) => (
          <ModelOnlyUpscaleFactorItem
            key={factor}
            image={image}
            label={t(
              factor === nativeScale
                ? 'widgets.gallery.itemActions.upscale.factorNative'
                : 'widgets.gallery.itemActions.upscale.scaleSuffix',
              { scale: factor }
            )}
            scale={factor === nativeScale ? 'native' : factor}
            upscaler={upscaler}
          />
        ))
      ) : (
        <>
          <ModelOnlyUpscaleFactorItem
            image={image}
            label={t('widgets.gallery.itemActions.upscale.nativeScaleUnknownFactor')}
            scale="native"
            upscaler={upscaler}
          />
          <MenuItem
            unavailableReason={t('widgets.upscale.nativeScaleUnknown')}
            value={`upscale-model-only-${upscaler.key}-other`}
          >
            <Text fontSize="md">{t('widgets.gallery.itemActions.upscale.otherFactors')}</Text>
          </MenuItem>
        </>
      )}
    </ContextSubMenu>
  );
};

const ModelOnlyUpscaleFactorItem = ({
  image,
  label,
  scale,
  upscaler,
}: {
  image: GalleryImage;
  label: string;
  scale: number | 'native';
  upscaler: SpandrelModelConfig;
}) => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const projectId = useActiveProjectId();
  const handleClick = useCallback(() => {
    const owner = captureAccountScope();

    // Loaded on pick: the menu is part of the editor's startup graph.
    void import('./modelOnlyUpscale')
      .then(({ queueModelOnlyUpscale }) =>
        queueModelOnlyUpscale({
          commands,
          image: { height: image.height, image_name: image.imageName, width: image.width },
          owner,
          projectId,
          scale,
          t,
          upscaleModel: upscaler,
        })
      )
      // The command reports its own failures, so this is its chunk failing to load; the pick must not go unanswered.
      .catch(() => {
        if (isAccountScopeCurrent(owner)) {
          commands.notifications.add({ kind: 'error', title: t('widgets.gallery.itemActions.upscale.loadFailed') });
        }
      });
  }, [commands, image.height, image.imageName, image.width, projectId, scale, t, upscaler]);

  return (
    <Menu.Item value={`upscale-model-only-${upscaler.key}-${scale}`} onClick={handleClick}>
      <Text fontSize="md" fontVariantNumeric="tabular-nums">
        {label}
      </Text>
    </Menu.Item>
  );
};
