/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { GenerateLora } from '@features/generation/core/types';
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';
import type { MouseEvent, ReactNode } from 'react';

import { Avatar, Badge, Box, Flex, Icon, Menu, Portal } from '@chakra-ui/react';
import { DEFAULT_LORA_WEIGHT_CONFIG, getDefaultLoraWeight } from '@features/generation/core/settings';
import { IconButton } from '@platform/ui/Button';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Tooltip } from '@platform/ui/Tooltip';
import { BoxIcon, ExternalLinkIcon, PowerIcon, PowerOffIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { createContext, memo, use, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerateFieldContextMenu } from './GenerateFieldContextMenu';
import { GenerateToggleSwitch } from './GenerateToggleSwitch';

const WEIGHT_MARKS = [-1, 0, 1, 2];

/** The model catalog's helpers a concept row needs; the caller supplies them through its own UI port. */
export interface ConceptModelPort {
  getBaseColorPalette(base: string): string;
  getBaseLabel(base: string): string;
  getImageUrl(key: string): string;
  /** Absent when this session may not manage models; the row then offers no manager link. */
  openInModelManager?: (key: string) => void;
}

export type ConceptUpdate = Partial<Pick<GenerateLora, 'isEnabled' | 'weight'>>;

export interface ConceptRowProps {
  /** False dims the row, badges it, and locks it off; the caller decides compatibility against its main model. */
  isCompatible?: boolean;
  /** `lora.weight` is the displayed weight, so a caller that defers commits passes its draft here. */
  lora: GenerateLora;
  models: ConceptModelPort;
  onRemove: (key: string) => void;
  /** Fires per weight step and on toggle; the caller owns any debouncing. */
  onUpdate: (key: string, update: ConceptUpdate) => void;
}

/** A removed row unmounts with keyboard focus inside it; the next (else previous) row's button takes it instead. */
const findNeighbourRow = (from: Element | null): HTMLElement | null => {
  const item = from?.closest('[role="listitem"]');

  for (const step of ['nextElementSibling', 'previousElementSibling'] as const) {
    for (let sibling = item?.[step]; sibling; sibling = sibling[step]) {
      const primary = sibling.querySelector<HTMLElement>('[data-list-primary]');

      if (primary) {
        return primary;
      }
    }
  }

  return null;
};

/** What the list's one context menu acts on, captured from the row that opened it. */
interface ConceptMenuTarget extends ListContextMenuAnchor {
  isActive: boolean;
  isAtDefault: boolean;
  isCompatible: boolean;
  key: string;
  onOpenInModelManager?: () => void;
  onRemove: () => void;
  onReset: () => void;
  onToggle: () => void;
}

interface ConceptMenuPort {
  open: (target: ConceptMenuTarget) => void;
  openKey: string | null;
}

const ConceptMenuContext = createContext<ConceptMenuPort | null>(null);

/**
 * The family's short list, with dividers between concepts. It owns the rows' single context menu: one menu per row
 * raced when a second row was right-clicked while the first's menu was closing, and both closed.
 */
export const ConceptList = ({ children, label }: { children: ReactNode; label: string }) => {
  const [target, setTarget] = useState<ConceptMenuTarget | null>(null);
  const port = useMemo<ConceptMenuPort>(() => ({ open: setTarget, openKey: target?.key ?? null }), [target]);
  const handleClose = useCallback(() => {
    target?.restoreFocus();
    setTarget(null);
  }, [target]);

  return (
    <ConceptMenuContext value={port}>
      <ListStack dividers label={label}>
        {children}
      </ListStack>
      {/* Keyed per row so switching rows replaces the menu rather than retargeting an open one. */}
      <ConceptContextMenu key={target?.key ?? 'closed'} target={target} onClose={handleClose} />
    </ConceptMenuContext>
  );
};

/**
 * One applied LoRA/concept: a list row (thumbnail, identity, toggle, remove, and a context menu) with its weight
 * scrubber as the row's detail, so the whole item shares one hover surface.
 */
export const ConceptRow = memo(function ConceptRow({
  isCompatible = true,
  lora,
  models,
  onRemove,
  onUpdate,
}: ConceptRowProps) {
  const { t } = useTranslation();
  const { key, name, trigger_phrases: triggerPhrases } = lora.model;
  const isActive = lora.isEnabled && isCompatible;
  const defaultWeight = getDefaultLoraWeight(lora.model);
  const menu = use(ConceptMenuContext);
  const handleToggle = useCallback((isEnabled: boolean) => onUpdate(key, { isEnabled }), [key, onUpdate]);
  const handleWeightChange = useCallback((weight: number) => onUpdate(key, { weight }), [key, onUpdate]);
  const removeFrom = useCallback(
    (origin: Element | null) => {
      const neighbour = findNeighbourRow(origin);

      onRemove(key);
      if (neighbour) {
        requestAnimationFrame(() => neighbour.focus());
      }
    },
    [key, onRemove]
  );
  const handleRemoveClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => removeFrom(event.currentTarget),
    [removeFrom]
  );
  const handleReset = useCallback(() => onUpdate(key, { weight: defaultWeight }), [defaultWeight, key, onUpdate]);
  const copyWeight = useCallback(() => String(lora.weight), [lora.weight]);
  const { openInModelManager } = models;
  const isAtDefault = lora.weight === defaultWeight;
  const handleContextMenu = (anchor: ListContextMenuAnchor) =>
    menu?.open({
      ...anchor,
      isActive,
      isAtDefault,
      isCompatible,
      key,
      onOpenInModelManager: openInModelManager ? () => openInModelManager(key) : undefined,
      onRemove: () => removeFrom(anchor.focusTarget()),
      onReset: handleReset,
      onToggle: () => handleToggle(!isActive),
    });

  return (
    <Box role="listitem">
      {/* Naming the group gives the row's "Weight" slider and controls their concept's context. */}
      <Box aria-label={name} opacity={isCompatible ? 1 : 0.68} role="group">
        <ListItem
          actions={
            <Flex gap={1}>
              <GenerateToggleSwitch
                checked={isActive}
                disabled={!isCompatible}
                label={
                  isActive
                    ? t('widgets.generate.disableConcept', { name })
                    : t('widgets.generate.enableConcept', { name })
                }
                onCheckedChange={handleToggle}
              />
              <Tooltip content={t('widgets.generate.removeConcept')}>
                <IconButton
                  aria-label={t('widgets.generate.removeConceptNamed', { name })}
                  color="fg.muted"
                  size="sm"
                  variant="ghost"
                  onClick={handleRemoveClick}
                >
                  <Trash2Icon />
                </IconButton>
              </Tooltip>
            </Flex>
          }
          badges={
            <>
              <Badge colorPalette={models.getBaseColorPalette(lora.model.base)} variant="surface">
                {models.getBaseLabel(lora.model.base)}
              </Badge>
              {isCompatible ? null : (
                <Badge colorPalette="orange" variant="surface">
                  {t('widgets.generate.incompatible')}
                </Badge>
              )}
            </>
          }
          description={triggerPhrases?.length ? triggerPhrases.join(', ') : undefined}
          detail={
            // Backed with the settings section's surface so the row's hover tint never washes out the track and fill.
            <Box bg="bg.muted" rounded="control">
              <GenerateFieldContextMenu
                copyValue={copyWeight}
                isAtDefault={lora.weight === defaultWeight}
                resetLabel={t('widgets.generate.resetToConceptDefault')}
                onReset={handleReset}
              >
                <ScrubberField
                  defaultValue={defaultWeight}
                  disabled={!isActive}
                  inputMax={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
                  inputMin={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
                  label={t('widgets.generate.weight')}
                  marks={WEIGHT_MARKS}
                  max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
                  min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
                  step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
                  value={lora.weight}
                  onChange={handleWeightChange}
                />
              </GenerateFieldContextMenu>
            </Box>
          }
          density="snug"
          leading={
            <Avatar.Root
              bg="bg.muted"
              borderColor="border"
              borderWidth="1px"
              color="fg.subtle"
              flexShrink={0}
              shape="rounded"
              size="xl"
            >
              {/* The fallback also covers a cover image that fails to load. */}
              <Avatar.Fallback>
                <Icon as={BoxIcon} boxSize="4" />
              </Avatar.Fallback>
              {lora.model.cover_image ? <Avatar.Image alt="" src={models.getImageUrl(key)} /> : null}
            </Avatar.Root>
          }
          isMenuOpen={menu?.openKey === key}
          role="presentation"
          title={name}
          onContextMenu={menu ? handleContextMenu : undefined}
        />
      </Box>
    </Box>
  );
});

const ConceptContextMenu = ({ onClose, target }: { onClose: () => void; target: ConceptMenuTarget | null }) => {
  const { t } = useTranslation();

  return (
    <Menu.Root
      lazyMount
      open={target !== null}
      positioning={{
        getAnchorRect: () => (target ? { height: 1, width: 1, x: target.x, y: target.y } : null),
        placement: 'bottom-start',
      }}
      unmountOnExit
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Menu.Positioner>
          {target ? (
            <MenuContent minW="12rem">
              {target.onOpenInModelManager ? (
                <MenuActionItem
                  icon={ExternalLinkIcon}
                  label={t('widgets.generate.conceptMenu.openInModelManager')}
                  value="open-in-model-manager"
                  onSelect={target.onOpenInModelManager}
                />
              ) : null}
              <MenuActionItem
                disabled={target.isAtDefault || !target.isActive}
                icon={RotateCcwIcon}
                label={t('widgets.generate.resetToConceptDefault')}
                value="reset-weight"
                onSelect={target.onReset}
              />
              <MenuActionItem
                disabled={!target.isCompatible}
                icon={target.isActive ? PowerOffIcon : PowerIcon}
                label={
                  target.isActive ? t('widgets.generate.conceptMenu.disable') : t('widgets.generate.conceptMenu.enable')
                }
                value="toggle"
                onSelect={target.onToggle}
              />
              <Menu.Separator />
              <MenuActionItem
                icon={Trash2Icon}
                label={t('widgets.generate.conceptMenu.remove')}
                tone="danger"
                value="remove"
                onSelect={target.onRemove}
              />
            </MenuContent>
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
