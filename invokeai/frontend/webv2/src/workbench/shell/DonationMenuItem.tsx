import { chakra, Icon, Menu } from '@chakra-ui/react';
import { apiFetchJson } from '@platform/transport/http';
import { type QueryClient, queryOptions, useQuery } from '@tanstack/react-query';
import { HeartIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface FrontendConfig {
  show_donation_link: boolean;
}

const frontendConfigQueryOptions = queryOptions({
  queryKey: ['frontend-config'],
  queryFn: ({ signal }) => apiFetchJson<FrontendConfig>('/api/v1/app/frontend_config', { signal }),
  staleTime: Infinity,
});

/**
 * Menu triggers call this on hover and focus so the setting usually resolves before the menu opens, rather than
 * inserting a row under the pointer. It also retries a failed request, which the mounted menu item never does.
 */
export const prefetchDonationMenuItem = (queryClient: QueryClient) =>
  queryClient.prefetchQuery(frontendConfigQueryOptions);

/** Both menus share the server's visibility policy and the same external destination. */
export const DonationMenuItem = () => {
  const { t } = useTranslation();
  const { data } = useQuery(frontendConfigQueryOptions);

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
