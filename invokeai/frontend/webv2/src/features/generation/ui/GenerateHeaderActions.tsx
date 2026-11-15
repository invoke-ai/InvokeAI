/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon, Menu, Portal } from '@chakra-ui/react';
import { isSupportedGenerateModel } from '@features/generation/core/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@features/generation/core/settings';
import { IconButton } from '@platform/ui/Button';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { RotateCcwIcon } from 'lucide-react';
import { lazy, Suspense, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GenerateResetSelection } from './GenerateResetMenuItems';

import { GeneratePresetsPopover } from './GeneratePresetsPopover';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import { GenerationSectionJumpMenu } from './shared/GenerationSectionJumpMenu';

const RESET_MENU_POSITIONING = { placement: 'bottom-end' } as const;

// The items' plan and wording load with the menu, off the editor's startup; hovering or focusing the button warms them.
const loadResetMenuItems = () => import('./GenerateResetMenuItems');
const GenerateResetMenuItems = lazy(() =>
  loadResetMenuItems().then((module) => ({ default: module.GenerateResetMenuItems }))
);
const preloadResetMenuItems = () => void loadResetMenuItems();

const ignoreSelect = () => undefined;

/** A disabled entry in the items' place, so the menu holds only menu items while they load. */
const MenuItemsLoading = () => {
  const { t } = useTranslation();

  return <MenuActionItem disabled label={t('common.loading')} value="loading" onSelect={ignoreSelect} />;
};
const MENU_ITEMS_FALLBACK = <MenuItemsLoading />;

export const GenerateHeaderActions = ({ instance }: { instance?: { id: string } }) => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const resetMenuIds = useTooltipTriggerIds();
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
      <Menu.Root ids={resetMenuIds} lazyMount positioning={RESET_MENU_POSITIONING} unmountOnExit>
        <Menu.Trigger asChild>
          <Tooltip content={label} ids={resetMenuIds}>
            <IconButton
              aria-label={label}
              color="fg.muted"
              disabled={!hasSelectedModel}
              size="sm"
              variant="ghost"
              onFocus={preloadResetMenuItems}
              onPointerEnter={preloadResetMenuItems}
            >
              <Icon as={RotateCcwIcon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent maxW="20rem">
              {/* The menu opens at once, so Escape and an outside press work while its items load. */}
              <Suspense fallback={MENU_ITEMS_FALLBACK}>
                <GenerateResetMenuItems resolve={resolve} />
              </Suspense>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </>
  );
};
