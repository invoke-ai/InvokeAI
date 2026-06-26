import type { NodePackInfo } from '@workbench/customNodes/api';

import { Badge, Box, Flex, HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, ConfirmDialog } from '@workbench/components/ui';
import { EmptyState } from '@workbench/components/ui/EmptyState';
import { useNodePackActions } from '@workbench/launchpad/nodes/shared/useNodePackActions';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSelector } from '@workbench/workflows/templates';
import { BlocksIcon, TriangleAlertIcon, Trash2Icon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { NodePreviewCard } from './NodePreviewCard';

/**
 * Detail pane for a selected pack: a metadata header with an uninstall action,
 * then a live preview of every node the pack contributes. Previews are built
 * from the invocation templates the backend parsed from its OpenAPI schema —
 * `pack.node_types` are the exact invocation type keys, and a pack's nodes all
 * share `template.nodePack === pack.name`.
 */
export const NodePackDetail = ({ onUninstalled, pack }: { onUninstalled: () => void; pack: NodePackInfo }) => {
  const status = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);

  useEffect(() => {
    ensureInvocationTemplatesLoaded();
  }, []);

  const packTemplates = useMemo(
    () => pack.node_types.map((nodeType) => templates[nodeType]).filter((template) => template !== undefined),
    [pack.node_types, templates]
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
              {pack.node_count} node{pack.node_count === 1 ? '' : 's'}
            </Badge>
          </HStack>
        </Stack>
        <UninstallButton onUninstalled={onUninstalled} pack={pack} />
      </HStack>

      <Stack gap="2">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="600" textTransform="uppercase">
          Nodes in this pack
        </Text>
        {isLoadingTemplates ? (
          <Flex align="center" justify="center" py="10">
            <Spinner color="fg.subtle" size="sm" />
          </Flex>
        ) : packTemplates.length === 0 ? (
          <EmptyState
            description="The backend has no node definitions for this pack yet. Reload nodes or restart InvokeAI to load them."
            icon={<Icon as={TriangleAlertIcon} />}
            title="No node previews available"
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
        Uninstall
      </Button>
      <ConfirmDialog
        body="Remove this pack from the custom nodes directory? A restart is required for removal to fully apply."
        confirmLabel="Uninstall Node Pack"
        isOpen={isConfirmOpen}
        title={`Uninstall ${pack.name}?`}
        onClose={() => setIsConfirmOpen(false)}
        onConfirm={handleUninstall}
      />
    </>
  );
};
