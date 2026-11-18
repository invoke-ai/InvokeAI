import type { UseCommandReturn } from '@chakra-ui/react';
import type { ReactNode, RefObject } from 'react';

import { Command, HStack, Icon, Kbd, Portal, Spacer, Text, chakra, useCombobox } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { EmptyState } from '@platform/ui/EmptyState';
import { ShortcutKeyGlyph } from '@workbench/hotkeys/keyGlyphs';
import { SearchIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { CommandPaletteRowsHandle } from './CommandPaletteRows';
import type { NavigablePaletteRow, PaletteEntry, PaletteSearchProvider, PaletteStage } from './entries';
import type { CommandPaletteController } from './useCommandPaletteController';

import {
  COMMAND_PALETTE_RESULTS_ID,
  COMMAND_PALETTE_RESULTS_LABEL_ID,
  CommandPaletteRows,
  getCommandPaletteRowDomId,
} from './CommandPaletteRows';
import { getCommandPaletteReturnFocusElement } from './paletteStore';
import { COMMAND_PALETTE_INPUT_ID, useCommandPaletteController } from './useCommandPaletteController';

const INPUT_PLACEHOLDER_STYLE = { color: 'fg.subtle' };
const INPUT_FOCUS_WITHIN_STYLE = { outlineColor: 'accent.focusRing' };
const DATE_HINT_ID = 'command-palette-date-hint';
const NO_PROVIDERS: PaletteSearchProvider[] = [];
const NAV_HINT_KEYS = ['arrowup', 'arrowdown'];
const ENTER_HINT_KEYS = ['enter'];
const ESC_HINT_KEYS = ['esc'];
const TAB_HINT_KEYS = ['tab'];
const COMBOBOX_IDS = {
  content: COMMAND_PALETTE_RESULTS_ID,
  input: COMMAND_PALETTE_INPUT_ID,
  item: getCommandPaletteRowDomId,
  label: COMMAND_PALETTE_RESULTS_LABEL_ID,
};
const NO_VALUE: string[] = [];
/** The palette's own status region announces results; the combobox's English one stays silent. */
const SILENT_RESULT_ANNOUNCEMENTS = { loading: '', noResults: '', resultsCount: () => '' };

/** Hints never wrap; a `shrink` hint truncates its label when a long translation leaves no room. */
const FooterHint = ({ children, keys, shrink = false }: { children: string; keys: string[]; shrink?: boolean }) => (
  <HStack flexShrink={shrink ? 1 : 0} gap="1" minW="0">
    {keys.map((key) => (
      <Kbd key={key} flexShrink={0} textTransform="lowercase">
        <ShortcutKeyGlyph fallback={key} part={key} />
      </Kbd>
    ))}
    <Text truncate>{children}</Text>
  </HStack>
);

const StagePreviewLifetime = ({ stage }: { stage: PaletteStage }) => {
  useMountEffect(() => stage.clearPreview);

  return null;
};

/**
 * Binds palette state to the command combobox, which stays open inline in the dialog. Palette state owns the query
 * and the highlight, where none means the first row, so the combobox's clears are ignored; row actions decide whether
 * the palette closes. Built on `useCombobox`, whose defaults keep what the palette needs (no auto-highlight, which
 * would leave a stage's current option on entry; arrows that wrap), where `useCommand` forces the opposite and cannot
 * silence the combobox's untranslated result announcements.
 */
const usePaletteCommand = (
  controller: CommandPaletteController,
  rowsRef: RefObject<CommandPaletteRowsHandle | null>
): UseCommandReturn<NavigablePaletteRow> => {
  const { collection, isBusy, onQueryChange, onRowActive, onSelectRow } = controller;
  const combobox = useCombobox({
    collection,
    disableLayer: true,
    highlightedValue: controller.activeRowId,
    ids: COMBOBOX_IDS,
    inputValue: controller.query,
    loading: isBusy,
    onHighlightChange: (details) => {
      if (details.highlightedValue) {
        onRowActive(details.highlightedValue);
      }
    },
    onInputValueChange: (details) => onQueryChange(details.inputValue),
    onSelect: (details) => onSelectRow(details.itemValue),
    open: true,
    scrollToIndexFn: ({ index }) => {
      const row = collection.at(index);

      if (row) {
        rowsRef.current?.scrollToRow(row.id);
      }
    },
    selectionBehavior: 'preserve',
    translations: SILENT_RESULT_ANNOUNCEMENTS,
    value: NO_VALUE,
  });

  return {
    combobox,
    command: undefined,
    loading: isBusy,
    selectItem: (row) => onSelectRow(row.id),
  };
};

export const CommandPaletteDialog = ({
  entries,
  isOpen,
  modifierKeyLabel,
  onClose,
  onExitComplete,
  providers = NO_PROVIDERS,
}: {
  entries: PaletteEntry[];
  isOpen: boolean;
  modifierKeyLabel: string;
  onClose: () => void;
  /** After the close animation; hosts that keep the palette mounted while it closes unmount it here. */
  onExitComplete?: () => void;
  providers?: PaletteSearchProvider[];
}) => {
  const onDialogOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Dialog.Root
      closeOnEscape={false}
      finalFocusEl={getCommandPaletteReturnFocusElement}
      lazyMount
      open={isOpen}
      placement="top"
      restoreFocus
      scrollBehavior="inside"
      unmountOnExit
      onExitComplete={onExitComplete}
      onOpenChange={onDialogOpenChange}
    >
      <CommandPaletteContent
        entries={entries}
        isOpen={isOpen}
        modifierKeyLabel={modifierKeyLabel}
        providers={providers}
        onClose={onClose}
      />
    </Dialog.Root>
  );
};

export default CommandPaletteDialog;

const CommandPaletteContent = ({
  entries,
  isOpen,
  modifierKeyLabel,
  onClose,
  providers,
}: {
  entries: PaletteEntry[];
  isOpen: boolean;
  modifierKeyLabel: string;
  onClose: () => void;
  providers: PaletteSearchProvider[];
}) => {
  const { t } = useTranslation();
  const controller = useCommandPaletteController({ entries, onClose, providers });
  const rowsRef = useRef<CommandPaletteRowsHandle>(null);
  const modEnterHintKeys = useMemo(() => [modifierKeyLabel, 'enter'], [modifierKeyLabel]);
  const command = usePaletteCommand(controller, rowsRef);

  let emptyState: ReactNode = null;

  if (controller.rows.length === 0) {
    if (controller.scopeIsError && controller.scopeLabel) {
      emptyState = (
        <EmptyState
          danger
          description={controller.scopeErrorMessage}
          py="6"
          title={t('commandPalette.states.couldNotSearch', { label: controller.scopeLabel })}
        >
          <Button variant="subtle" onClick={controller.onRetry}>
            {t('common.retry')}
          </Button>
        </EmptyState>
      );
    } else if (controller.scopeLabel && controller.scopeIsFetching) {
      emptyState = <EmptyState py="6" title={t('commandPalette.states.searching')} />;
    } else if (controller.scopeLabel && controller.trimmedQuery.length === 0) {
      emptyState = (
        <EmptyState py="6" title={t('commandPalette.states.noItemsYet', { label: controller.scopeLabel })} />
      );
    } else {
      emptyState = (
        <EmptyState py="6" title={t('commandPalette.states.noResults', { query: controller.trimmedQuery })} />
      );
    }
  }

  return (
    <Portal>
      {/* A preview ends when the palette closes, not when its exit animation finishes. */}
      {isOpen && controller.stage?.clearPreview ? <StagePreviewLifetime stage={controller.stage} /> : null}
      <Dialog.Backdrop bg="blackAlpha.300" />
      <Dialog.Positioner alignItems="flex-start" pt="15vh">
        <Dialog.Content
          maxW="none"
          overflow="hidden"
          overscrollBehavior="contain"
          p="0"
          w="min(640px, calc(100vw - 32px))"
          onKeyDown={controller.onContentKeyDown}
        >
          <Dialog.Title srOnly>{t('commandPalette.title')}</Dialog.Title>
          {/* Unstyled: the parts keep the palette's own layout; the root only adds a flex column. */}
          <Command.RootProvider display="flex" flexDirection="column" minH="0" unstyled value={command}>
            <Command.Control
              alignItems="center"
              borderBottomWidth="1px"
              borderColor="border.emphasized"
              // Inside the content's 1px border, so the focus outline follows its rounded top corners.
              borderTopRadius="calc({radii.l3} - 1px)"
              display="flex"
              flexShrink={0}
              gap="2.5"
              h="12"
              outline="2px solid transparent"
              outlineOffset="-2px"
              px="3"
              _focusWithin={INPUT_FOCUS_WITHIN_STYLE}
            >
              <Command.Indicator color="fg.muted" display="flex" flexShrink={0}>
                <Icon as={SearchIcon} boxSize="4" />
              </Command.Indicator>
              {controller.chipLabel ? (
                <chakra.button
                  alignItems="center"
                  aria-label={t('commandPalette.actions.closeScope', { label: controller.chipLabel })}
                  bg="bg.emphasized"
                  borderRadius="sm"
                  color="fg"
                  display="inline-flex"
                  flexShrink={0}
                  fontSize="md"
                  fontWeight="600"
                  gap="1"
                  px="1.5"
                  py="0.5"
                  title={t('commandPalette.actions.backToAll')}
                  type="button"
                  onClick={controller.onPopOverlay}
                >
                  {controller.chipLabel}
                  <Icon as={XIcon} boxSize="3" color="fg.muted" />
                </chakra.button>
              ) : null}
              <Command.Input
                autoFocus
                name="command-palette-query"
                type="search"
                aria-describedby={controller.dateInvalidHint ? DATE_HINT_ID : undefined}
                aria-invalid={controller.dateInvalidHint ? true : undefined}
                aria-label={t('commandPalette.inputLabel')}
                bg="transparent"
                color="fg"
                flex="1"
                fontSize="lg"
                outline="none"
                placeholder={controller.placeholder}
                w="full"
                _placeholder={INPUT_PLACEHOLDER_STYLE}
                onKeyDown={controller.onSearchKeyDown}
              />
              {controller.dateInvalidHint ? (
                <Text color="fg.error" flexShrink={0} fontSize="md" id={DATE_HINT_ID} maxW="45%" role="status" truncate>
                  {controller.dateInvalidHint}
                </Text>
              ) : controller.dateSummary ? (
                <Text
                  bg="bg.emphasized"
                  borderRadius="sm"
                  color="fg.muted"
                  flexShrink={0}
                  fontSize="md"
                  id={DATE_HINT_ID}
                  px="1.5"
                  py="0.5"
                  role="status"
                >
                  {controller.dateSummary}
                </Text>
              ) : null}
            </Command.Control>
            <Text aria-live="polite" role="status" srOnly>
              {controller.statusMessage}
            </Text>
            {controller.rows.length === 0 ? (
              emptyState
            ) : (
              <CommandPaletteRows
                ref={rowsRef}
                activeRowId={controller.activeRowId}
                isBusy={controller.isBusy}
                rows={controller.rows}
              />
            )}
            <Command.Footer
              alignItems="center"
              borderColor="border.emphasized"
              borderTopWidth="1px"
              color="fg.subtle"
              display="flex"
              flexShrink={0}
              fontSize="md"
              gap="4"
              h="8"
              hideBelow="sm"
              px="3"
            >
              <FooterHint keys={NAV_HINT_KEYS}>{t('commandPalette.footer.navigate')}</FooterHint>
              <FooterHint keys={ENTER_HINT_KEYS}>
                {controller.stage ? t('commandPalette.footer.pick') : t('commandPalette.footer.run')}
              </FooterHint>
              {controller.secondaryHint ? (
                <FooterHint keys={modEnterHintKeys} shrink>
                  {controller.secondaryHint}
                </FooterHint>
              ) : null}
              {controller.hasScopeRows || controller.activeRow?.kind === 'scope' ? (
                <FooterHint keys={TAB_HINT_KEYS}>{t('commandPalette.footer.scope')}</FooterHint>
              ) : null}
              <Spacer />
              <FooterHint keys={ESC_HINT_KEYS}>
                {controller.isOverlayOpen ? t('commandPalette.footer.back') : t('commandPalette.footer.close')}
              </FooterHint>
            </Command.Footer>
          </Command.RootProvider>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};
