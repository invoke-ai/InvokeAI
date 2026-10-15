/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { GenerateLora } from '@features/generation/core/types';
import type { MouseEvent, ReactNode } from 'react';

import { Avatar, Badge, Box, Flex, Icon, Menu, Portal } from '@chakra-ui/react';
import { DEFAULT_LORA_WEIGHT_CONFIG, getDefaultLoraWeight } from '@features/generation/core/settings';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { useRegisterDraftFlusher } from '@platform/react/draftRegistry';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { IconButton } from '@platform/ui/Button';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { MenuActionItem, MenuContent, useContextMenu } from '@platform/ui/Menu';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Tooltip } from '@platform/ui/Tooltip';
import { BoxIcon, ExternalLinkIcon, PowerIcon, PowerOffIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { createContext, memo, use, useCallback } from 'react';
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
  /** The committed concept. Weight edits stay in a local draft until idle or a workbench flush. */
  lora: GenerateLora;
  models: ConceptModelPort;
  onRemove: (key: string) => void;
  /** Weight commits are debounced; toggles commit immediately. Apply updates to the latest settings. */
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

interface ListSlot {
  container: Element;
  next: Element | null;
}

/** Where the list sat at each ancestor level, so focus can be placed after the last row unmounts it. */
const captureListSlots = (from: Element | null): ListSlot[] => {
  const slots: ListSlot[] = [];

  for (let node = from?.closest('[role="list"]'); node?.parentElement; node = node.parentElement) {
    slots.push({ container: node.parentElement, next: node.nextElementSibling });
  }

  return slots;
};

const TABBABLE = 'a[href], button, input:not([type="hidden"]), select, textarea, [tabindex], [contenteditable="true"]';

/**
 * The control before the removed list, read after the removal renders: the concept picker, which may only now be
 * enabled once the removed concept is no longer excluded from it.
 */
const findControlBeforeList = (slots: ListSlot[]): HTMLElement | null => {
  for (const { container, next } of slots) {
    if (!container.isConnected) {
      continue;
    }

    const preceding = [...container.querySelectorAll<HTMLElement>(TABBABLE)].filter(
      (element) =>
        element.tabIndex >= 0 &&
        !element.matches(':disabled') &&
        element.getClientRects().length > 0 &&
        (!next?.isConnected || Boolean(next.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_PRECEDING))
    );

    if (preceding.length > 0) {
      return preceding.at(-1) ?? null;
    }
  }

  return null;
};

const ConceptProjectContext = createContext<string | undefined>(undefined);

/** A short concept list shares its project lifetime with the rows' drafts and menus. */
export const ConceptList = ({
  children,
  label,
  projectId,
}: {
  children: ReactNode;
  label: string;
  projectId: string;
}) => (
  <ConceptProjectContext value={projectId}>
    <ListStack dividers label={label}>
      {children}
    </ListStack>
  </ConceptProjectContext>
);

/**
 * One applied LoRA/concept: a list row (thumbnail, identity, toggle, remove, and a context menu) with its weight
 * scrubber as the row's detail, so the whole item shares one hover surface.
 */
export const ConceptRow = memo(function ConceptRow(props: ConceptRowProps) {
  const projectId = use(ConceptProjectContext);
  const { lora, onUpdate } = props;
  const commitWeight = useCallback(
    (weight: number) => onUpdate(lora.model.key, { weight }),
    [lora.model.key, onUpdate]
  );
  const {
    draftValue: weight,
    flushDraftValue,
    setDraftValue,
  } = useDebouncedDraftValue({
    delayMs: 250,
    onCommit: commitWeight,
    resetKey: projectId,
    value: lora.weight,
  });
  useRegisterDraftFlusher(flushDraftValue);
  const update = useCallback(
    (key: string, update: ConceptUpdate) => {
      if (update.weight === undefined) {
        onUpdate(key, update);
      } else {
        setDraftValue(update.weight);
      }
    },
    [onUpdate, setDraftValue]
  );

  // Reset the menu on project changes without remounting/flushing the weight draft into the next project.
  return <ConceptRowContent key={projectId} {...props} weight={weight} onUpdate={update} />;
});

const ConceptRowContent = ({
  isCompatible = true,
  lora,
  models,
  onRemove,
  onUpdate,
  weight,
}: ConceptRowProps & { weight: number }) => {
  const { t } = useTranslation();
  const { key, name, trigger_phrases: triggerPhrases } = lora.model;
  const isActive = lora.isEnabled && isCompatible;
  const defaultWeight = getDefaultLoraWeight(lora.model);
  const menu = useContextMenu();
  // Kept through the exit animation: the anchor clears the moment the menu closes.
  const { release: releaseMenu, value: shownAnchor } = useExitRetainedValue(menu.anchor);
  const handleToggle = useCallback((isEnabled: boolean) => onUpdate(key, { isEnabled }), [key, onUpdate]);
  const handleWeightChange = useCallback((weight: number) => onUpdate(key, { weight }), [key, onUpdate]);
  const removeFrom = useCallback(
    (origin: Element | null) => {
      const neighbour = findNeighbourRow(origin);
      const slots = neighbour ? [] : captureListSlots(origin);

      onRemove(key);
      requestAnimationFrame(() => (neighbour ?? findControlBeforeList(slots))?.focus());
    },
    [key, onRemove]
  );
  const handleRemoveClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => removeFrom(event.currentTarget),
    [removeFrom]
  );
  const handleReset = useCallback(() => onUpdate(key, { weight: defaultWeight }), [defaultWeight, key, onUpdate]);
  const copyWeight = useCallback(() => String(weight), [weight]);
  const { openInModelManager } = models;

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
                isAtDefault={weight === defaultWeight}
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
                  value={weight}
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
          isMenuOpen={menu.anchor !== null}
          role="presentation"
          title={name}
          onContextMenu={menu.open}
        />
      </Box>
      <Menu.Root
        lazyMount
        open={menu.anchor !== null}
        positioning={{
          getAnchorRect: () => (shownAnchor ? { height: 1, width: 1, x: shownAnchor.x, y: shownAnchor.y } : null),
          placement: 'bottom-start',
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
                {openInModelManager ? (
                  <MenuActionItem
                    icon={ExternalLinkIcon}
                    label={t('widgets.generate.conceptMenu.openInModelManager')}
                    value="open-in-model-manager"
                    onSelect={() => openInModelManager(key)}
                  />
                ) : null}
                <MenuActionItem
                  disabled={weight === defaultWeight || !isActive}
                  icon={RotateCcwIcon}
                  label={t('widgets.generate.resetToConceptDefault')}
                  value="reset-weight"
                  onSelect={handleReset}
                />
                <MenuActionItem
                  disabled={!isCompatible}
                  icon={isActive ? PowerOffIcon : PowerIcon}
                  label={
                    isActive ? t('widgets.generate.conceptMenu.disable') : t('widgets.generate.conceptMenu.enable')
                  }
                  value="toggle"
                  onSelect={() => handleToggle(!isActive)}
                />
                <Menu.Separator />
                <MenuActionItem
                  icon={Trash2Icon}
                  label={t('widgets.generate.conceptMenu.remove')}
                  tone="danger"
                  value="remove"
                  onSelect={() => removeFrom(shownAnchor.focusTarget?.() ?? null)}
                />
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Box>
  );
};
