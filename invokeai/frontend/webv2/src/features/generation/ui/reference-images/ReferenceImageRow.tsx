/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerationIssue } from '@features/generation/core/generationIssues';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  GenerateReferenceImageConfig,
} from '@features/generation/core/types';
import type { TFunction } from 'i18next';
import type { ChangeEvent, CSSProperties, FocusEvent, KeyboardEvent, MouseEvent, ReactElement } from 'react';

import {
  Badge,
  Box,
  chakra,
  Flex,
  HStack,
  Icon,
  Input,
  Kbd,
  Menu,
  Portal,
  Stack,
  VisuallyHidden,
} from '@chakra-ui/react';
import { FindInGalleryButton } from '@features/gallery/mediaSlot';
import { isStrengthOnlyIPAdapterBase } from '@features/generation/core/baseGenerationPolicies';
import { getEffectiveReferenceImage } from '@features/generation/core/referenceImage';
import { DEFAULT_KREA2_STYLE_STRENGTH } from '@features/generation/core/settings';
import { getReferenceImageUrls } from '@features/generation/data/referenceImageUrls';
import { GenerationModelSelect as ModelSelect } from '@features/generation/ui/GenerationUiContext';
import { GenerateFieldContextMenu } from '@features/generation/ui/shared/GenerateFieldContextMenu';
import { type GenerateIssueAction, GenerateIssueButton } from '@features/generation/ui/shared/GenerateIssues';
import { GenerateRowSwitch } from '@features/generation/ui/shared/GenerateToggleSwitch';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { isImeComposing } from '@platform/browser/imeComposition';
import { useExitPresence, useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, IconButton } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { type ListContextMenuAnchor, ListItem } from '@platform/ui/list/ListItem';
import { MenuActionItem, MenuContent, useContextMenu } from '@platform/ui/Menu';
import { toaster } from '@platform/ui/toaster';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronDownIcon,
  CropIcon,
  CrosshairIcon,
  EllipsisIcon,
  ImageIcon,
  PencilIcon,
  PowerIcon,
  PowerOffIcon,
  RulerIcon,
  Trash2Icon,
} from 'lucide-react';
import { memo, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatWeight, getModeLabelKey } from './referenceImageConfig';
import {
  ClipVisionSelect,
  ControlSurface,
  DEFAULT_REFERENCE_WEIGHT,
  FluxReduxControls,
  IPAdapterSettings,
  IPAdapterWeightField,
  Krea2ReferenceModeControl,
  Krea2StyleReferenceControls,
} from './ReferenceImageControls';
import { ReferenceImageCropDialog } from './ReferenceImageCropDialog';

/** Reference image config variants that carry a selectable model. */
type ModelBackedConfig = Extract<
  GenerateReferenceImageConfig,
  { type: 'ip_adapter' | 'flux_redux' | 'flux_kontext_reference_image' }
>;

const PRIMARY_SELECTOR = '[data-list-primary]';
const FLUX_MODEL_TYPES = ['ip_adapter', 'flux_redux'];
const DEFAULT_MODEL_TYPES = ['ip_adapter'];

/** The model types a reference's picker offers on a base: adapters, and Redux on FLUX. */
export const getReferenceModelTypes = (base: string | undefined): string[] =>
  base === 'flux' ? FLUX_MODEL_TYPES : DEFAULT_MODEL_TYPES;
const COVER_IMG_STYLE: CSSProperties = { display: 'block', height: '100%', objectFit: 'cover', width: '100%' };
/** An off reference keeps its image recognizable but plainly not in use. */
const OFF_IMG_STYLE: CSSProperties = { ...COVER_IMG_STYLE, filter: 'grayscale(1) brightness(0.75)' };
const PROMPT_TAG_COPIED_MS = 1500;
const CHEVRON_OPEN_CSS = { transform: 'rotate(180deg)' } as const;
const MODEL_SELECT_SELECTOR = '[data-reference-model] button, [data-reference-model] input:not([type="hidden"])';
/** Issues fixed by choosing the reference's model; their popover offers that beside Turn off. */
const MODEL_ISSUE_KEYS = new Set(
  ['referenceImageNeedsIpAdapter', 'referenceImageNeedsRedux', 'referenceModelNotInstalled'].map(
    (key) => `widgets.generate.issues.${key}`
  )
);
const INDEX_PLACEHOLDER = '\u0000';

/**
 * Whether a name reads as some position's default ("Reference image 7"), in the active language and any case; null
 * when the locale's template cannot be matched safely (only the number, or a number formatter that drops the
 * placeholder), and then only a row's own default is recognized.
 */
export const getPositionalDefaultNameTest = (t: TFunction): ((name: string) => boolean) | null => {
  const parts = t('widgets.generate.referenceImageNumbered', { index: INDEX_PLACEHOLDER }).split(INDEX_PLACEHOLDER);
  const [before = '', after = ''] = parts.map((part) => part.trim());

  if (parts.length !== 2 || (before === '' && after === '')) {
    return null;
  }

  const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`^${escape(before)}\\s*\\d+\\s*${escape(after)}$`, 'iu');

  return (name) => pattern.test(name.trim());
};

const sameName = (left: string, right: string): boolean =>
  left.trim().toLocaleLowerCase() === right.trim().toLocaleLowerCase();

/** The families with a control on the row; the rest are a plain row whose image actions live in its menu. */
const hasRowControl = (config: GenerateReferenceImageConfig): boolean =>
  config.type === 'ip_adapter' ||
  config.type === 'flux_redux' ||
  config.type === 'flux_kontext_reference_image' ||
  config.type === 'krea2_reference_image' ||
  config.type === 'krea2_edit_reference_image';

/** The second line: what the reference does, read live from its settings, in the words its controls use. */
export const describeReferenceImage = (
  config: GenerateReferenceImageConfig,
  strengthOnly: boolean,
  t: TFunction
): string => {
  if (!config.image) {
    return t('widgets.generate.referenceImageNoImage');
  }

  const effective = getEffectiveReferenceImage(config.image);
  const size = t('widgets.generate.referenceImageSize', { height: effective.height, width: effective.width });
  const modelName = (model: { name: string } | null) => model?.name ?? t('widgets.generate.referenceImageNoModel');

  switch (config.type) {
    case 'ip_adapter': {
      const [begin, end] = config.beginEndStepPct;
      const range =
        begin === 0 && end === 1
          ? null
          : t('widgets.generate.stepRangeValue', { begin: Math.round(begin * 100), end: Math.round(end * 100) });

      return [strengthOnly ? null : t(getModeLabelKey(config.method)), range, modelName(config.model)]
        .filter(Boolean)
        .join(' · ');
    }
    case 'flux_redux':
      return modelName(config.model);
    case 'flux_kontext_reference_image':
      return size;
    case 'krea2_reference_image':
      return t('widgets.generate.modeStyle');
    case 'krea2_edit_reference_image':
      return `${t('widgets.generate.modeEdit')} · ${size}`;
    default:
      return config.image.crop ? `${size} · ${t('widgets.generate.referenceImageCropped')}` : size;
  }
};

export interface ReferenceImageRowProps {
  count: number;
  /**
   * An installed model the row's picker could offer. Without one, an issue asking for a model offers the Model
   * Manager instead of a picker with nothing in it.
   */
  hasModelChoice?: boolean;
  /** Opens Add Models filtered to a model type, for an issue no installed model fixes. */
  onOpenModelManager?: (modelType: string) => void;
  index: number;
  /**
   * False lists the row dimmed, badged and inert: the selected model takes no references, and the list above it says
   * so with its one way out.
   */
  isSupported?: boolean;
  /** How the prompt names this reference, for models that address references by tag. */
  promptTag?: string;
  referenceImage: GenerateReferenceImage;
  selectedModel: GenerateModelConfig | undefined;
  onFindInGallery: (imageName: string) => void;
  onMove: (id: string, direction: -1 | 1) => void;
  onPatch: (id: string, patch: Partial<GenerateReferenceImage>) => void;
  onRemove: (id: string) => void;
  /** A name that is empty, or reads as a positional default (`isDefaultName`), clears the custom name. */
  onRename: (id: string, name: string, isDefaultName: (name: string) => boolean) => void;
  onUseSize: (image: GenerateReferenceImageAsset) => void;
}

/**
 * One reference image as a list row, the same construction as a concept row: its image tile, name and live summary,
 * a switch and a menu, with its one main control on the row and the rest of its settings under a disclosure. The
 * row's button opens the menu (Enter, Space, right-click); F2 or a double-click renames; Right and Left open and
 * close the settings; Alt+Up and Alt+Down move it; Delete removes it.
 */
const ReferenceImageRowBase = ({
  count,
  hasModelChoice = true,
  onOpenModelManager,
  index,
  isSupported = true,
  onFindInGallery,
  onMove,
  onPatch,
  onRemove,
  onRename,
  onUseSize,
  promptTag,
  referenceImage,
  selectedModel,
}: ReferenceImageRowProps) => {
  const { t } = useTranslation();
  const { config, id, isEnabled } = referenceImage;
  const image = config.image;
  const selectedBase = selectedModel?.base;
  const strengthOnly = isStrengthOnlyIPAdapterBase(selectedBase);
  const defaultName = t('widgets.generate.referenceImageNumbered', { index: index + 1 });
  const name = referenceImage.name ?? defaultName;
  const hasSettings = isSupported && hasRowControl(config);
  // Closed until asked for: the chevron, Right, or an issue's Choose model.
  const [isOpen, setIsOpen] = useState(false);
  const showSettings = hasSettings && isOpen;
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameStartDefault, setRenameStartDefault] = useState(defaultName);
  const [isCropOpen, setIsCropOpen] = useState(false);
  const cropDialog = useExitPresence(isCropOpen);
  const menu = useContextMenu();
  // Rename from the menu: the menu closing must not hand focus back to the row, which would end the rename at once.
  const keepFocusOnClose = useRef(false);
  const openMenu = useCallback(
    (anchor: ListContextMenuAnchor) => {
      keepFocusOnClose.current = false;
      menu.open({
        ...anchor,
        restoreFocus: () => {
          if (!keepFocusOnClose.current) {
            anchor.restoreFocus();
          }
        },
      });
    },
    [menu]
  );
  // Kept through the exit animation: the anchor clears the moment the menu closes.
  const { release: releaseMenu, value: shownAnchor } = useExitRetainedValue(menu.anchor);
  const rowRef = useRef<HTMLDivElement>(null);
  const settingsId = useId();
  const canMoveUp = index > 0;
  const canMoveDown = index < count - 1;
  const originalImageName = image?.original.image.image_name;

  /** Moving or re-rendering the row's button can drop focus; it goes back to the row once the change has rendered. */
  const focusRowSoon = useCallback(() => {
    requestAnimationFrame(() => rowRef.current?.querySelector<HTMLElement>(PRIMARY_SELECTOR)?.focus());
  }, []);
  const changeConfig = useCallback(
    (nextConfig: GenerateReferenceImageConfig) => onPatch(id, { config: nextConfig }),
    [id, onPatch]
  );
  // A Krea-2 mode switch moves the mode control between the row (Edit) and its settings (Style): the settings open so
  // it stays in view, and keyboard focus follows it to its new place.
  const changeKrea2Mode = useCallback(
    (nextConfig: GenerateReferenceImageConfig) => {
      const group = document.activeElement?.closest('[data-scope="segment-group"]');
      const hadFocus = group !== null && group !== undefined && rowRef.current?.contains(group) === true;

      if (nextConfig.type === 'krea2_reference_image') {
        setIsOpen(true);
      }

      changeConfig(nextConfig);

      if (hadFocus) {
        requestAnimationFrame(() =>
          rowRef.current
            ?.querySelector<HTMLElement>('[data-scope="segment-group"] input[type="radio"]:checked')
            ?.focus()
        );
      }
    },
    [changeConfig]
  );
  const setEnabled = useCallback((nextEnabled: boolean) => onPatch(id, { isEnabled: nextEnabled }), [id, onPatch]);
  const move = useCallback(
    (direction: -1 | 1) => {
      if (direction === -1 ? !canMoveUp : !canMoveDown) {
        return;
      }

      onMove(id, direction);
      focusRowSoon();
    },
    [canMoveDown, canMoveUp, focusRowSoon, id, onMove]
  );
  // The next reference takes focus, else the previous one, else the add row: never the page.
  const remove = useCallback(() => {
    const row = rowRef.current;
    const list = row?.closest('[role="list"]');
    const rows = list ? [...list.querySelectorAll<HTMLElement>('[data-reference-image-id]')] : [];
    const at = row ? rows.indexOf(row) : -1;
    const neighbour = rows[at + 1] ?? rows[at - 1];
    const target =
      neighbour?.querySelector<HTMLElement>(PRIMARY_SELECTOR) ??
      list?.querySelector<HTMLElement>('[data-reference-add]');

    onRemove(id);
    requestAnimationFrame(() => target?.focus());
  }, [id, onRemove]);
  const startRename = useCallback(() => {
    if (isSupported) {
      setRenameStartDefault(defaultName);
      setIsRenaming(true);
    }
  }, [defaultName, isSupported]);
  const renameFromMenu = useCallback(() => {
    keepFocusOnClose.current = true;
    startRename();
  }, [startRename]);
  const endRename = useCallback(
    (refocus: boolean) => {
      setIsRenaming(false);

      if (refocus) {
        focusRowSoon();
      }
    },
    [focusRowSoon]
  );
  // This row's default stores no name, so it keeps following the row's position. The default when editing began
  // counts too: a reorder from another tab while the field is open must not store the untouched text as a name.
  const isPositionalDefault = useMemo(() => getPositionalDefaultNameTest(t), [t]);
  const [renameError, setRenameError] = useState<string | null>(null);
  const renameErrorId = useId();
  // Another position's default would make two rows read alike: the field says so and keeps the name it was given.
  const validateName = useCallback(
    (draft: string): string | null => {
      const name = draft.trim();

      return name !== '' &&
        !sameName(name, defaultName) &&
        !sameName(name, renameStartDefault) &&
        isPositionalDefault?.(name) === true
        ? t('widgets.generate.referenceImageNameReserved', { name })
        : null;
    },
    [defaultName, isPositionalDefault, renameStartDefault, t]
  );
  const commitName = useCallback(
    (draft: string, refocus: boolean) => {
      setRenameError(null);
      endRename(refocus);
      onRename(id, draft, (name) => sameName(name, defaultName) || sameName(name, renameStartDefault));
    },
    [defaultName, endRename, id, onRename, renameStartDefault]
  );
  const cancelRename = useCallback(
    (refocus: boolean) => {
      setRenameError(null);
      endRename(refocus);
    },
    [endRename]
  );
  const openCrop = useCallback(() => setIsCropOpen(true), []);
  const closeCrop = useCallback(() => setIsCropOpen(false), []);
  const handleCrop = useCallback(
    (croppedImage: GenerateReferenceImageAsset) => changeConfig({ ...config, image: croppedImage }),
    [changeConfig, config]
  );
  const useSize = useCallback(() => {
    if (image) {
      onUseSize(image);
    }
  }, [image, onUseSize]);
  // Reveal original media; crop intermediates have no gallery cell.
  const findInGallery = useCallback(() => {
    if (originalImageName !== undefined) {
      onFindInGallery(originalImageName);
    }
  }, [onFindInGallery, originalImageName]);
  const openMenuFrom = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const button = event.currentTarget;
      const rect = button.getBoundingClientRect();

      openMenu({
        focusTarget: () => (button.isConnected ? button : null),
        restoreFocus: () => button.focus(),
        x: rect.right,
        y: rect.bottom,
      });
    },
    [openMenu]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement;

      // Only the row's own button: its controls and fields keep their keys.
      if (
        !isSupported ||
        !target.matches(PRIMARY_SELECTOR) ||
        target.closest('[data-reference-image-id]') !== rowRef.current
      ) {
        return;
      }

      const plain = !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey;

      if (event.key === 'F2' && plain) {
        startRename();
      } else if (event.altKey && !event.ctrlKey && !event.metaKey && event.key === 'ArrowUp') {
        move(-1);
      } else if (event.altKey && !event.ctrlKey && !event.metaKey && event.key === 'ArrowDown') {
        move(1);
      } else if (plain && event.key === 'ArrowRight' && hasSettings) {
        setIsOpen(true);
      } else if (plain && event.key === 'ArrowLeft' && hasSettings) {
        setIsOpen(false);
      } else if (plain && event.key === 'Delete') {
        // A held key repeats: only the first press removes, so holding Delete cannot empty the list.
        if (!event.repeat) {
          remove();
        }
      } else {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
    },
    [hasSettings, isSupported, move, remove, startRename]
  );
  const handleDoubleClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      const primary = (event.target as HTMLElement).closest(PRIMARY_SELECTOR);

      if (!isRenaming && primary && primary.closest('[data-reference-image-id]') === rowRef.current) {
        startRename();
      }
    },
    [isRenaming, startRename]
  );
  const toggleSettings = useCallback(() => setIsOpen((open) => !open), []);
  // An issue fixed by choosing a model: open the settings that hold the picker and focus it once it has rendered and
  // the closing popover has handed focus back.
  const chooseModel = useCallback(() => {
    setIsOpen(true);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => rowRef.current?.querySelector<HTMLElement>(MODEL_SELECT_SELECTOR)?.focus())
    );
  }, []);
  const modelType = config.type === 'flux_redux' ? 'flux_redux' : 'ip_adapter';
  const chooseModelAction = useMemo<GenerateIssueAction | null>(
    () =>
      hasModelChoice
        ? { label: t('widgets.generate.referenceImageChooseModel'), onSelect: chooseModel }
        : onOpenModelManager
          ? { label: t('widgets.generate.openModelManager'), onSelect: () => onOpenModelManager(modelType) }
          : null,
    [chooseModel, hasModelChoice, modelType, onOpenModelManager, t]
  );
  const getIssueAction = useCallback(
    (issue: GenerationIssue) => (issue.messageKey && MODEL_ISSUE_KEYS.has(issue.messageKey) ? chooseModelAction : null),
    [chooseModelAction]
  );

  const modelSelect = (modelConfig: ModelBackedConfig) => (
    <ReferenceModelSelect
      config={modelConfig}
      disabled={!isEnabled}
      selectedBase={selectedBase}
      strengthOnly={strengthOnly}
      onConfigChange={changeConfig}
    />
  );
  const weightMenu = (label: string, value: number, defaultValue: number, onReset: () => void, field: ReactElement) => (
    <GenerateFieldContextMenu
      copyValue={() => formatWeight(value)}
      fieldLabel={label}
      isAtDefault={value === defaultValue || !isEnabled}
      resetLabel={t('widgets.generate.resetToDefault')}
      onReset={onReset}
    >
      {field}
    </GenerateFieldContextMenu>
  );

  let rowControl: ReactElement | null = null;
  let settings: ReactElement | null = null;

  switch (config.type) {
    case 'ip_adapter':
      rowControl = weightMenu(
        strengthOnly ? t('widgets.generate.strength') : t('widgets.generate.weight'),
        config.weight,
        DEFAULT_REFERENCE_WEIGHT,
        () => changeConfig({ ...config, weight: DEFAULT_REFERENCE_WEIGHT }),
        <IPAdapterWeightField
          config={config}
          disabled={!isEnabled}
          strengthOnly={strengthOnly}
          onChange={changeConfig}
        />
      );
      settings = (
        <IPAdapterSettings
          key={config.model?.key ?? 'no-model'}
          config={config}
          disabled={!isEnabled}
          strengthOnly={strengthOnly}
          onChange={changeConfig}
        >
          {modelSelect(config)}
        </IPAdapterSettings>
      );
      break;
    case 'flux_redux':
      rowControl = (
        <FluxReduxControls config={config} disabled={!isEnabled} labelPlacement="inside" onChange={changeConfig} />
      );
      settings = modelSelect(config);
      break;
    case 'flux_kontext_reference_image':
      rowControl = modelSelect(config);
      break;
    case 'krea2_reference_image':
      rowControl = weightMenu(
        t('widgets.generate.styleStrength'),
        config.styleStrength,
        DEFAULT_KREA2_STYLE_STRENGTH,
        () => changeConfig({ ...config, styleStrength: DEFAULT_KREA2_STYLE_STRENGTH }),
        <Krea2StyleReferenceControls config={config} disabled={!isEnabled} onChange={changeConfig} />
      );
      settings = <Krea2ReferenceModeControl config={config} onChange={changeKrea2Mode} />;
      break;
    case 'krea2_edit_reference_image':
      rowControl = <Krea2ReferenceModeControl config={config} onChange={changeKrea2Mode} />;
      break;
    default:
      break;
  }

  const imageActions = image ? (
    <HStack gap="0.5">
      <IconButton
        aria-label={t('widgets.generate.cropReferenceImageNamed', { name })}
        color="fg.muted"
        disabled={!isEnabled}
        size="sm"
        tooltip={t('common.crop')}
        variant="ghost"
        onClick={openCrop}
      >
        <CropIcon />
      </IconButton>
      <IconButton
        aria-label={t('widgets.generate.useSizeOfReferenceImage', { name })}
        color="fg.muted"
        disabled={!isEnabled || !selectedModel}
        size="sm"
        tooltip={t('widgets.generate.useSize')}
        variant="ghost"
        onClick={useSize}
      >
        <RulerIcon />
      </IconButton>
      <FindInGalleryButton color="fg.muted" name={originalImageName} variant="ghost" onFind={findInGallery} />
    </HStack>
  ) : null;

  const detail = hasSettings ? (
    <Stack gap="1.5">
      <HStack gap="1" minW="0">
        {/* The Krea-2 mode control backs its own segments under its label. */}
        {config.type === 'krea2_edit_reference_image' ? (
          <Box flex="1" minW="0">
            {rowControl}
          </Box>
        ) : (
          <ControlSurface>{rowControl}</ControlSurface>
        )}
        <IconButton
          aria-controls={settingsId}
          aria-expanded={showSettings}
          aria-label={t('widgets.generate.referenceImageSettings', { name })}
          color="fg.muted"
          size="sm"
          tooltip={t('widgets.generate.referenceImageSettings', { name })}
          variant="ghost"
          onClick={toggleSettings}
        >
          <Box
            asChild
            css={showSettings ? CHEVRON_OPEN_CSS : undefined}
            transition="transform var(--wb-motion-duration-fast)"
          >
            <ChevronDownIcon />
          </Box>
        </IconButton>
      </HStack>
      {showSettings ? (
        <Stack gap="1.5" id={settingsId}>
          {settings}
          {imageActions}
        </Stack>
      ) : null}
    </Stack>
  ) : undefined;

  return (
    <Box
      ref={rowRef}
      data-reference-image-id={id}
      role="listitem"
      onDoubleClick={handleDoubleClick}
      onKeyDown={handleKeyDown}
    >
      {/* Naming the group gives the row's controls their reference's context ("Weight", "Hero pose"). An off row
          stays at full opacity, so it is still readable; its text recedes and its tile greys. */}
      <Box aria-label={name} color={isEnabled ? undefined : 'fg.subtle'} role="group">
        <ListItem
          actions={
            <Flex align="center" gap="1">
              {promptTag ? <PromptTagButton tag={promptTag} /> : null}
              {/* A reference that blocks Invoke says so beside its switch, with the fix one press away. */}
              <GenerateIssueButton
                field="referenceImage"
                getAction={getIssueAction}
                referenceImageId={id}
                section="guidance"
              />
              <GenerateRowSwitch
                checked={isEnabled}
                label={
                  isEnabled
                    ? t('widgets.generate.disableReferenceImageNamed', { name })
                    : t('widgets.generate.enableReferenceImageNamed', { name })
                }
                onCheckedChange={setEnabled}
              />
              <IconButton
                aria-expanded={menu.anchor !== null}
                aria-haspopup="menu"
                aria-label={t('widgets.generate.referenceImageMoreActions', { name })}
                color="fg.muted"
                size="sm"
                tooltip={t('widgets.generate.referenceImageMoreActions', { name })}
                variant="ghost"
                onClick={openMenuFrom}
              >
                <EllipsisIcon />
              </IconButton>
            </Flex>
          }
          badges={
            isSupported ? undefined : (
              <Badge colorPalette="orange" flexShrink={0} variant="surface">
                {t('widgets.generate.incompatible')}
              </Badge>
            )
          }
          density="snug"
          description={
            renameError ? (
              <chakra.span color="fg.error" display="block" id={renameErrorId} minW="0" role="alert">
                {renameError}
              </chakra.span>
            ) : (
              <chakra.span color={isEnabled ? 'fg.muted' : 'fg.subtle'} display="block" minW="0" truncate>
                {describeReferenceImage(config, strengthOnly, t)}
              </chakra.span>
            )
          }
          detail={detail}
          isDisabled={!isSupported}
          isMenuOpen={menu.anchor !== null}
          leading={<ReferenceImageTile image={image} isEnabled={isEnabled} />}
          role="presentation"
          title={name}
          titleEditor={
            isRenaming ? (
              <ReferenceImageNameField
                errorId={renameError ? renameErrorId : undefined}
                initialName={name}
                validate={validateName}
                onCancel={cancelRename}
                onCommit={commitName}
                onErrorChange={setRenameError}
              />
            ) : undefined
          }
          titleTruncate="end"
          onContextMenu={openMenu}
        />
      </Box>
      <Menu.Root
        lazyMount
        open={menu.anchor !== null}
        positioning={{
          getAnchorRect: () => (shownAnchor ? { height: 1, width: 1, x: shownAnchor.x, y: shownAnchor.y } : null),
          placement: 'bottom-end',
        }}
        unmountOnExit
        onExitComplete={releaseMenu}
        onOpenChange={(event) => {
          if (!event.open) {
            menu.close();
          }
        }}
        onRequestDismiss={menu.onRequestDismiss}
      >
        <Portal>
          <Menu.Positioner>
            {shownAnchor ? (
              <MenuContent minW="12rem">
                <MenuActionItem
                  hint={t('widgets.generate.referenceImageMenu.renameHint')}
                  icon={PencilIcon}
                  label={t('widgets.generate.referenceImageMenu.rename')}
                  // F2 reads the same on every platform, so it needs none of the workbench's key formatting.
                  shortcut={<Kbd textTransform="lowercase">F2</Kbd>}
                  value="rename"
                  onSelect={renameFromMenu}
                />
                <MenuActionItem
                  disabled={!image || !isEnabled}
                  icon={CropIcon}
                  label={t('widgets.generate.referenceImageMenu.crop')}
                  value="crop"
                  onSelect={openCrop}
                />
                <MenuActionItem
                  disabled={!image || !isEnabled || !selectedModel}
                  icon={RulerIcon}
                  label={t('widgets.generate.useSize')}
                  value="use-size"
                  onSelect={useSize}
                />
                <MenuActionItem
                  disabled={!image}
                  icon={CrosshairIcon}
                  label={t('widgets.gallery.findInGallery')}
                  value="find-in-gallery"
                  onSelect={findInGallery}
                />
                <Menu.Separator />
                <MenuActionItem
                  disabled={!canMoveUp}
                  icon={ArrowUpIcon}
                  label={t('widgets.generate.referenceImageMenu.moveUp')}
                  value="move-up"
                  onSelect={() => move(-1)}
                />
                <MenuActionItem
                  disabled={!canMoveDown}
                  icon={ArrowDownIcon}
                  label={t('widgets.generate.referenceImageMenu.moveDown')}
                  value="move-down"
                  onSelect={() => move(1)}
                />
                <MenuActionItem
                  icon={isEnabled ? PowerOffIcon : PowerIcon}
                  label={
                    isEnabled
                      ? t('widgets.generate.referenceImageMenu.turnOff')
                      : t('widgets.generate.referenceImageMenu.turnOn')
                  }
                  value="toggle"
                  onSelect={() => setEnabled(!isEnabled)}
                />
                <Menu.Separator />
                <MenuActionItem
                  icon={Trash2Icon}
                  label={t('widgets.generate.referenceImageMenu.remove')}
                  tone="danger"
                  value="remove"
                  onSelect={remove}
                />
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {image && cropDialog.isMounted ? (
        <ReferenceImageCropDialog
          key={cropDialog.generation}
          image={image}
          isOpen={cropDialog.isOpen}
          onApply={handleCrop}
          onClose={closeCrop}
          onExitComplete={cropDialog.release}
        />
      ) : null}
    </Box>
  );
};

export const ReferenceImageRow = memo(ReferenceImageRowBase);

/** The image as it will be used (the crop, if cropped), marked when cropped and greyed while the reference is off. */
const ReferenceImageTile = ({
  image,
  isEnabled,
}: {
  image: GenerateReferenceImageAsset | null;
  isEnabled: boolean;
}) => {
  const urls = image ? getReferenceImageUrls(image) : null;

  return (
    <Box
      alignItems="center"
      bg="bg.muted"
      borderColor="border"
      borderWidth="1px"
      boxSize="9"
      color="fg.subtle"
      display="flex"
      flexShrink={0}
      justifyContent="center"
      overflow="hidden"
      position="relative"
      rounded="sm"
    >
      {urls ? (
        <img alt="" draggable={false} src={urls.thumbnailUrl} style={isEnabled ? COVER_IMG_STYLE : OFF_IMG_STYLE} />
      ) : (
        <Icon as={ImageIcon} boxSize="4" />
      )}
      {image?.crop ? (
        <Box
          alignItems="center"
          bg="bg/80"
          bottom="0"
          color="fg"
          data-cropped=""
          display="flex"
          insetInlineEnd="0"
          justifyContent="center"
          position="absolute"
          roundedTopLeft="sm"
          boxSize="3.5"
        >
          <Icon as={CropIcon} boxSize="2.5" />
        </Box>
      ) : null}
    </Box>
  );
};

/**
 * Renames in the title's place, as a layer row does: Enter or moving focus away commits, Escape cancels, and Enter or
 * Escape hand focus back to the row. A name left empty, or set to the default, clears the custom name.
 */
const ReferenceImageNameField = ({
  errorId,
  initialName,
  onCancel,
  onCommit,
  onErrorChange,
  validate,
}: {
  /** The message saying why the name cannot be kept, shown in the row's second line while it applies. */
  errorId: string | undefined;
  initialName: string;
  onCancel: (refocus: boolean) => void;
  onCommit: (name: string, refocus: boolean) => void;
  onErrorChange: (error: string | null) => void;
  /** Why a draft cannot be kept, or null. Enter keeps the field open on it; leaving the field keeps the old name. */
  validate: (draft: string) => string | null;
}) => {
  const { t } = useTranslation();
  const input = useRef<HTMLInputElement | null>(null);
  // Refocusing the row blurs the still-mounted field, which would end it a second time; the first end claims it.
  const isDone = useRef(false);
  const finish = useCallback(
    (refocus: boolean, cancelled: boolean) => {
      if (isDone.current) {
        return;
      }

      const draft = input.current?.value ?? '';
      const invalid = !cancelled && validate(draft) !== null;

      isDone.current = true;

      if (cancelled || invalid) {
        onCancel(refocus);
      } else {
        onCommit(draft, refocus);
      }
    },
    [onCancel, onCommit, validate]
  );
  const handleChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => onErrorChange(validate(event.currentTarget.value)),
    [onErrorChange, validate]
  );
  const handleBlur = useCallback(
    (event: FocusEvent<HTMLInputElement>) => {
      const next = event.relatedTarget;

      // A menu that just closed (the row menu's Rename) takes focus back to its content while it animates out; the
      // field keeps it instead.
      if (
        next instanceof HTMLElement &&
        next.closest('[data-scope="menu"][data-part="content"][data-state="closed"]')
      ) {
        event.currentTarget.focus();
        return;
      }

      // Leaving the window keeps the field open; it commits when focus moves elsewhere in the page.
      if (next === null && !document.hasFocus()) {
        return;
      }

      finish(next === null, false);
    },
    [finish]
  );
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      event.stopPropagation();

      // While an IME composes, Enter confirms a candidate and Escape drops it; neither ends the rename.
      if (isImeComposing(event.nativeEvent)) {
        return;
      }

      if (event.key === 'Enter') {
        event.preventDefault();
        const error = validate(event.currentTarget.value);

        // A name that cannot be kept stays in the field with its reason, rather than changing on its own.
        if (error) {
          onErrorChange(error);
          return;
        }

        finish(true, false);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        finish(true, true);
      }
    },
    [finish, onErrorChange, validate]
  );
  const focusOnMount = useCallback((element: HTMLInputElement | null) => {
    input.current = element;
    element?.focus();
    element?.select();
  }, []);
  const stopPropagation = useCallback((event: { stopPropagation: () => void }) => event.stopPropagation(), []);

  return (
    <Input
      ref={focusOnMount}
      aria-describedby={errorId}
      aria-invalid={errorId ? true : undefined}
      aria-label={t('common.rename')}
      defaultValue={initialName}
      flex="1"
      fontSize="md"
      fontWeight="600"
      h="5"
      minW="0"
      mx="-1"
      px="1"
      size="xs"
      userSelect="text"
      onBlur={handleBlur}
      onChange={handleChange}
      onClick={stopPropagation}
      onDoubleClick={stopPropagation}
      onKeyDown={handleKeyDown}
      onPointerDown={stopPropagation}
    />
  );
};

/** The model a reference runs through, with the CLIP Vision encoder beside it for an SD/FLUX IP-Adapter. */
const ReferenceModelSelect = ({
  config,
  disabled,
  onConfigChange,
  selectedBase,
  strengthOnly,
}: {
  config: ModelBackedConfig;
  disabled: boolean;
  selectedBase: string | undefined;
  strengthOnly: boolean;
  onConfigChange: (config: GenerateReferenceImageConfig) => void;
}) => {
  const { t } = useTranslation();
  const modelTypes = getReferenceModelTypes(selectedBase);
  const filterModel = useCallback((model: ModelConfig) => model.base === selectedBase, [selectedBase]);
  const selectReferenceModel = useCallback(
    (model: ModelConfig | null) => {
      if (!model || model.base !== selectedBase) {
        return;
      }

      if (model.type === 'flux_redux') {
        onConfigChange({ image: config.image, imageInfluence: 'highest', model, type: 'flux_redux' });
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
    <HStack gap="1.5" minW="0">
      <ControlSurface data-reference-model="">
        <Field disabled={disabled} label={t('widgets.generate.model')} labelPlacement="inside">
          <ModelSelect
            disabled={disabled}
            filter={filterModel}
            // A Kontext reference may switch to Redux on FLUX, as it always could.
            modelTypes={modelTypes}
            placeholder={t('widgets.generate.selectModel')}
            value={config.model?.key ?? null}
            onChange={selectReferenceModel}
          />
        </Field>
      </ControlSurface>
      {config.type === 'ip_adapter' && !strengthOnly ? (
        <Box bg="bg.subtle" data-control-surface="" flexShrink={0} rounded="control">
          <ClipVisionSelect
            key={config.model?.key ?? 'no-model'}
            config={config}
            disabled={disabled}
            onChange={onConfigChange}
          />
        </Box>
      ) : null}
    </HStack>
  );
};

const PromptTagButton = ({ tag }: { tag: string }) => {
  const { t } = useTranslation();
  const [hasCopied, setHasCopied] = useState(false);
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useMountEffect(() => () => {
    if (resetTimerRef.current !== null) {
      clearTimeout(resetTimerRef.current);
    }
  });

  const copyTag = useCallback(() => {
    copyText(tag)
      .then(() => {
        setHasCopied(true);
        if (resetTimerRef.current !== null) {
          clearTimeout(resetTimerRef.current);
        }
        resetTimerRef.current = setTimeout(() => setHasCopied(false), PROMPT_TAG_COPIED_MS);
      })
      .catch(() =>
        toaster.create({ description: t('common.clipboardBlocked'), title: t('common.couldNotCopy'), type: 'error' })
      );
  }, [t, tag]);

  const canCopy = canCopyText();
  // Without a clipboard the button shows its own tooltip, the reason, so the explanation joins it there rather than
  // opening a second tooltip; it leaves out the hint's "click to copy", which is untrue then.
  const unavailableReason = canCopy
    ? undefined
    : `${t('widgets.generate.referenceImagePromptTagExplanation', { tag })} ${t('common.copyUnavailable')}`;
  const button = (
    <Button
      aria-label={t('widgets.generate.copyReferenceImagePromptTag', { tag })}
      color="fg.muted"
      flexShrink="0"
      fontFamily="mono"
      fontWeight="normal"
      size="xs"
      unavailableReason={unavailableReason}
      variant="subtle"
      onClick={copyTag}
    >
      {hasCopied ? <Icon as={CheckIcon} color="green.solid" /> : null}
      {tag}
      <VisuallyHidden aria-live="polite">
        {hasCopied ? t('widgets.generate.referenceImagePromptTagCopied', { tag }) : ''}
      </VisuallyHidden>
    </Button>
  );

  return canCopy ? (
    <Tooltip content={t('widgets.generate.referenceImagePromptTagHint', { tag })}>{button}</Tooltip>
  ) : (
    button
  );
};
