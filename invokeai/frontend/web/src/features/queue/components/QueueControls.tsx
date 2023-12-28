import { Flex, Spacer } from '@chakra-ui/react';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import ProgressBar from 'features/system/components/ProgressBar';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

import { InvokeQueueBackButton } from './InvokeQueueBackButton';
import { QueueActionsMenuButton } from './QueueActionsMenuButton';

const QueueControls = () => {
  const isPrependEnabled = useFeatureStatus('prependQueue').isFeatureEnabled;
  return (
    <Flex
      sx={{
        w: 'full',
        position: 'relative',
        borderRadius: 'base',
        gap: 2,
        flexDir: 'column',
      }}
    >
      <InvButtonGroup size="lg" isAttached={false}>
        {isPrependEnabled && <QueueFrontButton />}
        <InvokeQueueBackButton />
        <Spacer />
        <QueueActionsMenuButton />
        {/* <CancelCurrentQueueItemButton asIconButton />
        {isResumeEnabled && <ResumeProcessorButton asIconButton />}
        {isPauseEnabled && <PauseProcessorButton asIconButton />} */}
        <ClearQueueButton asIconButton />
      </InvButtonGroup>
      <Flex h={1} w="full">
        <ProgressBar />
      </Flex>
    </Flex>
  );
};

export default QueueControls;

// const QueueCounts = () => {
//   const { t } = useTranslation();
//   const dispatch = useAppDispatch();
//   const { hasItems, pending } = useGetQueueStatusQuery(undefined, {
//     selectFromResult: ({ data }) => {
//       if (!data) {
//         return {
//           hasItems: false,
//           pending: 0,
//         };
//       }

//       const { pending, in_progress } = data.queue;

//       return {
//         hasItems: pending + in_progress > 0,
//         pending,
//       };
//     },
//   });

//   const handleClick = useCallback(() => {
//     dispatch(setActiveTab('queue'));
//   }, [dispatch]);

//   return (
//     <Flex
//       justifyContent="space-between"
//       alignItems="center"
//       pe={1}
//       data-testid="queue-count"
//     >
//       <Spacer />
//       <InvButton
//         onClick={handleClick}
//         size="sm"
//         variant="link"
//         opacity={0.7}
//         fontStyle="oblique 10deg"
//       >
//         {hasItems
//           ? t('queue.queuedCount', {
//               pending,
//             })
//           : t('queue.queueEmpty')}
//       </InvButton>
//     </Flex>
//   );
// };
