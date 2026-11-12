import { Icon, Menu, Portal } from '@chakra-ui/react';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { TableOfContentsIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { revealPanelSection, usePanelSections } from './sectionPanel';

const JUMP_MENU_POSITIONING = { placement: 'bottom-end' } as const;

/**
 * A widget header's list of its panel's sections, in order. Choosing one opens it when folded, brings it to the top
 * of the panel and focuses its title. Hidden while the panel shows fewer than two sections.
 */
export const GenerationSectionJumpMenu = ({ panelId }: { panelId: string | undefined }) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const sections = usePanelSections(panelId);
  const ui = useGenerationUi();
  const handleSelect = useCallback(
    ({ value }: { value: string }) => {
      const section = sections.find((entry) => entry.key === value);

      if (section) {
        revealPanelSection(section, ui.sectionPreferences.setSectionsOpen);
      }
    },
    [sections, ui]
  );

  if (sections.length < 2) {
    return null;
  }

  const label = t('widgets.generate.jumpToSection');

  return (
    <Menu.Root ids={ids} positioning={JUMP_MENU_POSITIONING} onSelect={handleSelect}>
      <Menu.Trigger asChild>
        <Tooltip content={label} ids={ids}>
          <IconButton aria-label={label} color="fg.muted" size="sm" variant="ghost">
            <Icon as={TableOfContentsIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent>
            {sections.map((section) => (
              <Menu.Item key={section.key} value={section.key}>
                {section.label}
              </Menu.Item>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
