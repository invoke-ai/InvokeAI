import type { ReactNode } from 'react';

import { Dialog, HStack, Icon, Kbd, Portal, Spacer, Text, chakra } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { SearchIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { CommandPaletteRowsHandle } from './CommandPaletteRows';
import type { PaletteEntry, PaletteSearchProvider, PaletteStage } from './entries';

import { CommandPaletteRows, getCommandPaletteRowDomId } from './CommandPaletteRows';
import { getCommandPaletteReturnFocusElement } from './paletteStore';
import { COMMAND_PALETTE_INPUT_ID, useCommandPaletteController } from './useCommandPaletteController';

const INPUT_PLACEHOLDER_STYLE = { color: 'fg.subtle' };
const INPUT_FOCUS_WITHIN_STYLE = { outlineColor: 'accent.focusRing' };
const DATE_HINT_ID = 'command-palette-date-hint';
const NO_PROVIDERS: PaletteSearchProvider[] = [];
const NAV_HINT_KEYS = ['↑', '↓'];
const ENTER_HINT_KEYS = ['↵'];
const ESC_HINT_KEYS = ['esc'];
const TAB_HINT_KEYS = ['tab'];

const FooterHint = ({ children, keys }: { children: string; keys: string[] }) => (
  <HStack gap="1">
    {keys.map((key) => (
      <Kbd key={key} size="sm" textTransform="lowercase">
        {key}
      </Kbd>
    ))}
    <Text>{children}</Text>
  </HStack>
);

const StagePreviewLifetime = ({ stage }: { stage: PaletteStage }) => {
  useMountEffect(() => stage.clearPreview);

  return null;
};

export const CommandPaletteDialog = ({
  entries,
  isOpen,
  modifierKeyLabel,
  onClose,
  providers = NO_PROVIDERS,
}: {
  entries: PaletteEntry[];
  isOpen: boolean;
  modifierKeyLabel: string;
  onClose: () => void;
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
      restoreFocus
      scrollBehavior="inside"
      unmountOnExit
      onOpenChange={onDialogOpenChange}
    >
      {isOpen ? (
        <CommandPaletteContent
          entries={entries}
          modifierKeyLabel={modifierKeyLabel}
          providers={providers}
          onClose={onClose}
        />
      ) : null}
    </Dialog.Root>
  );
};

export default CommandPaletteDialog;

const CommandPaletteContent = ({
  entries,
  modifierKeyLabel,
  onClose,
  providers,
}: {
  entries: PaletteEntry[];
  modifierKeyLabel: string;
  onClose: () => void;
  providers: PaletteSearchProvider[];
}) => {
  const { t } = useTranslation();
  const controller = useCommandPaletteController({ entries, onClose, providers });
  const rowsRef = useRef<CommandPaletteRowsHandle>(null);
  const modEnterHintKeys = useMemo(() => [modifierKeyLabel, '↵'], [modifierKeyLabel]);
  const handleSearchKeyDown = controller.onSearchKeyDown;
  const onSearchKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) =>
      handleSearchKeyDown(event, (index) => rowsRef.current?.scrollToIndex(index)),
    [handleSearchKeyDown]
  );

  let emptyState: ReactNode = null;

  if (controller.rows.length === 0) {
    if (controller.scopeIsError && controller.scopeLabel) {
      emptyState = (
        <EmptyState danger py="6" title={t('commandPalette.states.couldNotSearch', { label: controller.scopeLabel })}>
          <Button size="xs" variant="subtle" onClick={controller.onRetry}>
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
      {controller.stage?.clearPreview ? <StagePreviewLifetime stage={controller.stage} /> : null}
      <Dialog.Backdrop bg="blackAlpha.300" />
      <Dialog.Positioner alignItems="flex-start" pt="15vh">
        <Dialog.Content
          maxW="none"
          overflow="hidden"
          overscrollBehavior="contain"
          p="0"
          w="min(560px, calc(100vw - 32px))"
          onKeyDown={controller.onContentKeyDown}
        >
          <Dialog.Title srOnly>{t('commandPalette.title')}</Dialog.Title>
          <HStack
            borderBottomWidth="1px"
            borderColor="border.emphasized"
            flexShrink={0}
            gap="2.5"
            h="12"
            outline="2px solid transparent"
            outlineOffset="-2px"
            px="3"
            _focusWithin={INPUT_FOCUS_WITHIN_STYLE}
          >
            <Icon as={SearchIcon} boxSize="4" color="fg.muted" flexShrink={0} />
            {controller.chipLabel ? (
              <chakra.button
                alignItems="center"
                aria-label={t('commandPalette.actions.closeScope', { label: controller.chipLabel })}
                bg="bg.emphasized"
                borderRadius="sm"
                color="fg"
                cursor="pointer"
                display="inline-flex"
                flexShrink={0}
                fontSize="xs"
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
            <chakra.input
              autoComplete="off"
              autoFocus
              id={COMMAND_PALETTE_INPUT_ID}
              name="command-palette-query"
              spellCheck={false}
              type="search"
              aria-activedescendant={
                controller.activeRowId ? getCommandPaletteRowDomId(controller.activeRowId) : undefined
              }
              aria-controls="command-palette-results"
              aria-describedby={controller.dateInvalidHint ? DATE_HINT_ID : undefined}
              aria-expanded="true"
              aria-invalid={controller.dateInvalidHint ? true : undefined}
              aria-label={t('commandPalette.inputLabel')}
              bg="transparent"
              color="fg"
              flex="1"
              fontSize="sm"
              outline="none"
              placeholder={controller.placeholder}
              role="combobox"
              value={controller.query}
              w="full"
              _placeholder={INPUT_PLACEHOLDER_STYLE}
              onChange={controller.onSearchChange}
              onKeyDown={onSearchKeyDown}
            />
            {controller.dateInvalidHint ? (
              <Text color="fg.error" flexShrink={0} fontSize="xs" id={DATE_HINT_ID} maxW="45%" role="status" truncate>
                {controller.dateInvalidHint}
              </Text>
            ) : controller.dateSummary ? (
              <Text
                bg="bg.emphasized"
                borderRadius="sm"
                color="fg.muted"
                flexShrink={0}
                fontSize="xs"
                id={DATE_HINT_ID}
                px="1.5"
                py="0.5"
                role="status"
              >
                {controller.dateSummary}
              </Text>
            ) : null}
          </HStack>
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
              onActive={controller.onRowActive}
              onRun={controller.onRowRun}
            />
          )}
          <HStack
            borderColor="border.emphasized"
            borderTopWidth="1px"
            color="fg.subtle"
            flexShrink={0}
            fontSize="xs"
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
              <FooterHint keys={modEnterHintKeys}>{controller.secondaryHint}</FooterHint>
            ) : null}
            {controller.hasScopeRows || controller.activeRow?.kind === 'scope' ? (
              <FooterHint keys={TAB_HINT_KEYS}>{t('commandPalette.footer.scope')}</FooterHint>
            ) : null}
            <Spacer />
            <FooterHint keys={ESC_HINT_KEYS}>
              {controller.isOverlayOpen ? t('commandPalette.footer.back') : t('commandPalette.footer.close')}
            </FooterHint>
          </HStack>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};
