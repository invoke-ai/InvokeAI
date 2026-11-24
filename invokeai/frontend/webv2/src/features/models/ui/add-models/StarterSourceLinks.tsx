import type { MouseEvent } from 'react';

/* oxlint-disable react-perf/jsx-no-new-object-as-prop -- Menu positioning is a declarative configuration. */
import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { isSafeUrl } from '@features/models/core/modelSettingsIO';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { ChevronDownIcon, ExternalLinkIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

// Source links keep the browser menu instead of opening the enclosing row’s install menu.
const keepNativeLinkMenu = (event: MouseEvent<HTMLAnchorElement>) => event.stopPropagation();

export const StarterSourceLinks = ({ name, urls }: { name: string; urls: readonly string[] | undefined }) => {
  const { t } = useTranslation();
  const sources = [...new Set(urls?.filter(isSafeUrl) ?? [])];
  const label = t('models.visitModelSource', { name });

  if (!sources.length) {
    return null;
  }

  if (sources.length === 1) {
    return (
      <Button asChild size="sm" variant="ghost">
        <a aria-label={label} href={sources[0]} rel="noreferrer" target="_blank" onContextMenu={keepNativeLinkMenu}>
          <Icon as={ExternalLinkIcon} boxSize="3" />
          {t('models.visitSource')}
        </a>
      </Button>
    );
  }

  return (
    <Menu.Root lazyMount positioning={{ placement: 'bottom-end' }} unmountOnExit>
      <Menu.Trigger asChild>
        <Button aria-label={label} size="sm" variant="ghost">
          <Icon as={ExternalLinkIcon} boxSize="3" />
          {t('models.visitSource')}
          <Icon as={ChevronDownIcon} boxSize="3" />
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent maxW="min(32rem, 90vw)">
            {sources.map((url) => (
              <Menu.Item key={url} asChild value={url}>
                <a href={url} rel="noreferrer" target="_blank" onContextMenu={keepNativeLinkMenu}>
                  <Icon as={ExternalLinkIcon} boxSize="3" flexShrink={0} />
                  <Text overflowWrap="anywhere">{url.replace(/^https?:\/\//, '')}</Text>
                </a>
              </Menu.Item>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
