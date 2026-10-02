import { chakra, Icon, Menu } from '@chakra-ui/react';
import { apiFetchJson } from '@platform/transport/http';
import { useQuery } from '@tanstack/react-query';
import { HeartIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface FrontendConfig {
  show_donation_link: boolean;
}

/** Both menus share the server's visibility policy and the same external destination. */
export const DonationMenuItem = () => {
  const { t } = useTranslation();
  const { data } = useQuery({
    queryKey: ['frontend-config'],
    queryFn: ({ signal }) => apiFetchJson<FrontendConfig>('/api/v1/app/frontend_config', { signal }),
    staleTime: Infinity,
  });

  // Keep the optional link hidden until the deployment's preference is known, including on request failure.
  if (!data?.show_donation_link) {
    return null;
  }

  return (
    <Menu.Item asChild value="donate">
      <chakra.a href="https://github.com/sponsors/invoke-ai" rel="noreferrer" target="_blank">
        <Icon as={HeartIcon} boxSize="3.5" />
        <Menu.ItemText>{t('common.donateToInvokeAI')}</Menu.ItemText>
      </chakra.a>
    </Menu.Item>
  );
};
