/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';

import { Badge, Box, Flex, HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { useNodePackActions } from '@features/nodes/ui/shared/useNodePackActions';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSelector } from '@features/workflow/react';
import { Button, ConfirmDialog } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { BlocksIcon, TriangleAlertIcon, Trash2Icon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { NodePreviewCard } from './NodePreviewCard';

/**
 * Detail pane for a selected pack: a metadata header with an uninstall action,
 * then a live preview of every node the pack contributes. Previews are built
 * from the invocation templates the backend parsed from its OpenAPI schema —
 * `pack.nodeTypes` are the exact invocation type keys, and a pack's nodes all
 * share `template.nodePack === pack.name`.
 */
export const NodePackDetail = ({ onUninstalled, pack }: { onUninstalled: () => void; pack: NodePackInfo }) => {
  const { t } = useTranslation();
  const status = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);

  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  const packTemplates = useMemo(
    () => pack.nodeTypes.map((nodeType) => templates[nodeType]).filter((template) => template !== undefined),
    [pack.nodeTypes, templates]
  );

  const isLoadingTemplates = (status === 'idle' || status === 'loading') && packTemplates.length === 0;

  return (
    <Stack gap="4" pb="4">
      <HStack align="start" gap="3" justify="space-between">
        <Stack flex="1" gap="1.5" minW="0">
          <HStack gap="2" minW="0">
            <Icon as={BlocksIcon} boxSize="4" color="fg.muted" flexShrink={0} />
            <Text fontSize="sm" fontWeight="700" minW="0" truncate>
              {pack.name}
            </Text>
          </HStack>
          <Text color="fg.subtle" fontFamily="mono" fontSize="2xs" overflowWrap="anywhere">
            {pack.path}
          </Text>
          <HStack gap="1.5" wrap="wrap">
            <Badge colorPalette="blue" fontSize="2xs" variant="surface">
              {t('nodes.nodeCount', { count: pack.nodeCount })}
            </Badge>
          </HStack>
        </Stack>
        <UninstallButton onUninstalled={onUninstalled} pack={pack} />
      </HStack>

      <Stack gap="2">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="600" textTransform="uppercase">
          {t('nodes.nodesInPack')}
        </Text>
        {isLoadingTemplates ? (
          <Flex align="center" justify="center" py="10">
            <Spinner color="fg.subtle" size="sm" />
          </Flex>
        ) : packTemplates.length === 0 ? (
          <EmptyState
            description={t('nodes.noPreviewsDescription')}
            icon={<Icon as={TriangleAlertIcon} />}
            title={t('nodes.noPreviews')}
          />
        ) : (
          <Flex gap="8" wrap="wrap" px="2">
            {packTemplates.map((template) => (
              <Box key={template.type} flexShrink={0} w="18rem">
                <NodePreviewCard template={template} />
              </Box>
            ))}
          </Flex>
        )}
      </Stack>
    </Stack>
  );
};

const UninstallButton = ({ onUninstalled, pack }: { onUninstalled: () => void; pack: NodePackInfo }) => {
  const { t } = useTranslation();
  const { uninstall } = useNodePackActions();
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [isUninstalling, setIsUninstalling] = useState(false);

  const handleUninstall = async () => {
    setIsUninstalling(true);

    try {
      await uninstall(pack, onUninstalled);
    } finally {
      setIsUninstalling(false);
    }
  };

  return (
    <>
      <Button
        colorPalette="red"
        flexShrink={0}
        loading={isUninstalling}
        size="xs"
        variant="outline"
        onClick={() => setIsConfirmOpen(true)}
      >
        <Icon as={Trash2Icon} boxSize="3" />
        {t('nodes.uninstall')}
      </Button>
      <ConfirmDialog
        body={t('nodes.uninstallBody')}
        confirmLabel={t('nodes.uninstallPack')}
        isOpen={isConfirmOpen}
        title={t('nodes.uninstallTitle', { name: pack.name })}
        onClose={() => setIsConfirmOpen(false)}
        onConfirm={handleUninstall}
      />
    </>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
