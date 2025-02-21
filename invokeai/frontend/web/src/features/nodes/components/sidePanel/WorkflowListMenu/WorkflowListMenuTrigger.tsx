import {
  Box,
  Button,
  Flex,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowName } from 'features/nodes/store/workflowSlice';
import UploadWorkflowButton from 'features/workflowLibrary/components/UploadWorkflowButton';
import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFolderOpenFill } from 'react-icons/pi';

import { WorkflowList } from './WorkflowList';
import { WorkflowSearch } from './WorkflowSearch';
import { WorkflowSortControl } from './WorkflowSortControl';

export const WorkflowListMenuTrigger = () => {
  const workflowListMenu = useWorkflowListMenu();
  const { t } = useTranslation();
  const workflowCategories = useStore($workflowCategories);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const workflowName = useAppSelector(selectWorkflowName);

  return (
    <Popover
      isOpen={workflowListMenu.isOpen}
      onClose={workflowListMenu.close}
      onOpen={workflowListMenu.open}
      isLazy
      lazyBehavior="unmount"
      placement="bottom-end"
      initialFocusRef={searchInputRef}
    >
      <PopoverTrigger>
        <Button variant="ghost" rightIcon={<PiFolderOpenFill />} size="sm">
          <Text
            display="auto"
            noOfLines={1}
            overflow="hidden"
            textOverflow="ellipsis"
            whiteSpace="nowrap"
            wordBreak="break-all"
          >
            {workflowName || t('workflows.chooseWorkflowFromLibrary')}
          </Text>
        </Button>
      </PopoverTrigger>
      <Portal>
        <PopoverContent p={4} w={512} maxW="full" minH={512} maxH="full">
          <PopoverBody flex="1 1 0">
            <Flex w="full" h="full" flexDir="column" gap={2}>
              <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
                <WorkflowSearch searchInputRef={searchInputRef} />
                <WorkflowSortControl />
                <UploadWorkflowButton />
              </Flex>
              <Box position="relative" w="full" h="full">
                <ScrollableContent>
                  {workflowCategories.map((category) => (
                    <WorkflowList key={category} category={category} />
                  ))}
                </ScrollableContent>
              </Box>
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
};
