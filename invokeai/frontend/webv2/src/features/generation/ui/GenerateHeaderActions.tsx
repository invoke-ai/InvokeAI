/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon, Menu, Portal } from '@chakra-ui/react';
import { isSupportedGenerateModel } from '@features/generation/core/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@features/generation/core/settings';
import { IconButton } from '@platform/ui/Button';
import { createMenuPartLoader, MenuContent, MenuPartPlaceholder, useMenuPart } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { RotateCcwIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateResetSelection } from './GenerateResetMenuItems';

import { GeneratePresetsPopover } from './GeneratePresetsPopover';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import { GenerationSectionJumpMenu } from './shared/GenerationSectionJumpMenu';

const RESET_MENU_POSITIONING = { placement: 'bottom-end' } as const;

// The items' plan and wording load on demand, off the editor's startup; pointing at or focusing the button warms them.
const resetMenuItems = createMenuPartLoader(() =>
  import('./GenerateResetMenuItems').then((module) => module.GenerateResetMenuItems)
);
// Also on open: assistive technology can press the button without pointing at or focusing it first.
const onResetMenuOpenChange = ({ open }: { open: boolean }) => {
  if (open) {
    resetMenuItems.load();
  }
};

export const GenerateHeaderActions = ({ instance }: { instance?: { id: string } }) => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const resetMenuIds = useTooltipTriggerIds();
  const items = useMenuPart(resetMenuItems);
  const models = ui.models.catalog;
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const resolve = (values: Record<string, unknown>): GenerateResetSelection | null => {
    const settings = normalizeGenerateSettings(values);
    const selectedModel = settings && supportedModels.find((model) => model.key === settings.modelKey);

    return settings && selectedModel ? { selectedModel, settings } : null;
  };
  // The button needs only whether a model is selected; the menu's items read the rest while it shows.
  const hasSelectedModel = useGenerateValues((values) => resolve(values) !== null);

  const label = t('widgets.generate.resetSettings');

  return (
    <>
      <GenerationSectionJumpMenu panelId={instance?.id} />
      <GeneratePresetsPopover />
      <Menu.Root
        ids={resetMenuIds}
        lazyMount
        positioning={RESET_MENU_POSITIONING}
        unmountOnExit
        onOpenChange={onResetMenuOpenChange}
      >
        <Menu.Trigger asChild>
          <Tooltip content={label} ids={resetMenuIds}>
            <IconButton
              aria-label={label}
              color="fg.muted"
              disabled={!hasSelectedModel}
              size="sm"
              variant="ghost"
              // Capture: the Tooltip hands the menu trigger's `onFocus` to the button over its own.
              onFocusCapture={resetMenuItems.load}
              onPointerEnter={resetMenuItems.load}
            >
              <Icon as={RotateCcwIcon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent aria-busy={items.status === 'loading'} maxW="20rem">
              {items.status === 'loaded' ? (
                <items.part resolve={resolve} />
              ) : (
                <MenuPartPlaceholder status={items.status} />
              )}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </>
  );
};
